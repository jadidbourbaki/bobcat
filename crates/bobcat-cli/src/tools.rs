//! Tool calls in the form LFM2 writes them.
//!
//! LFM2 writes tool calls between its tool call tokens as a Python list of calls, as in
//! `[get_weather(city='Paris', days=3)]`. Keyword arguments hold Python literals: quoted
//! strings, numbers, `True`, `False`, `None`, and lists and dicts. When a system prompt asks for
//! JSON, the model writes a JSON list of `{"name", "arguments"}` objects in place of the
//! Python list, so the parser accepts both.

use serde_json::{Map, Value};

/// One call of a tool with its arguments.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ToolCall {
    /// The tool's name.
    pub(crate) name: String,
    /// The arguments, by name.
    pub(crate) arguments: Map<String, Value>,
}

/// Return the tool calls in `text`, the model's output between its tool call tokens.
pub(crate) fn parse(text: &str) -> Result<Vec<ToolCall>, String> {
    let text = text.trim();
    if let Ok(Value::Array(calls)) = serde_json::from_str::<Value>(text) {
        return calls.into_iter().map(json_call).collect();
    }
    let mut parser = Parser { text, pos: 0 };
    let calls = parser.calls()?;
    parser.skip_space();
    if parser.pos < text.len() {
        return Err(format!(
            "unexpected text after the tool calls: {}",
            &text[parser.pos..]
        ));
    }
    Ok(calls)
}

/// Return the tool call a JSON object `{"name", "arguments"}` describes.
fn json_call(call: Value) -> Result<ToolCall, String> {
    let Value::Object(mut call) = call else {
        return Err("a JSON tool call must be an object".to_owned());
    };
    let Some(Value::String(name)) = call.remove("name") else {
        return Err("a JSON tool call needs a name".to_owned());
    };
    let arguments = match call.remove("arguments") {
        Some(Value::Object(arguments)) => arguments,
        None => Map::new(),
        Some(_) => return Err(format!("the arguments of {name} must be an object")),
    };
    Ok(ToolCall { name, arguments })
}

/// Return `calls` in the form LFM2 writes them, for replaying earlier calls in a conversation.
pub(crate) fn format(calls: &[ToolCall]) -> String {
    let calls: Vec<String> = calls
        .iter()
        .map(|call| {
            let arguments: Vec<String> = call
                .arguments
                .iter()
                .map(|(name, value)| format!("{name}={}", literal(value)))
                .collect();
            format!("{}({})", call.name, arguments.join(", "))
        })
        .collect();
    format!("[{}]", calls.join(", "))
}

/// Return `value` as a Python literal, with dicts and lists in JSON, as LFM2's template writes
/// arguments.
fn literal(value: &Value) -> String {
    match value {
        Value::String(text) => {
            let escaped = text
                .replace('\\', "\\\\")
                .replace('\'', "\\'")
                .replace('\n', "\\n")
                .replace('\r', "\\r");
            format!("'{escaped}'")
        }
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::Null => "None".to_owned(),
        Value::Number(number) => number.to_string(),
        Value::Array(_) | Value::Object(_) => value.to_string(),
    }
}

/// A cursor over the Python text of a list of tool calls.
struct Parser<'t> {
    text: &'t str,
    pos: usize,
}

impl<'t> Parser<'t> {
    fn rest(&self) -> &'t str {
        &self.text[self.pos..]
    }

    fn skip_space(&mut self) {
        let trimmed = self.rest().trim_start();
        self.pos = self.text.len() - trimmed.len();
    }

    /// Consume `expected` after any space, and report whether it was there.
    fn eat(&mut self, expected: char) -> bool {
        self.skip_space();
        if self.rest().starts_with(expected) {
            self.pos += expected.len_utf8();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, expected: char) -> Result<(), String> {
        if self.eat(expected) {
            Ok(())
        } else {
            Err(format!("expected {expected:?} at {:?}", self.rest()))
        }
    }

    /// Parse a list of calls. The brackets are optional, since a model may drop them for one
    /// call.
    fn calls(&mut self) -> Result<Vec<ToolCall>, String> {
        let bracketed = self.eat('[');
        let mut calls = Vec::new();
        loop {
            self.skip_space();
            if bracketed && self.eat(']') {
                return Ok(calls);
            }
            if !bracketed && self.rest().is_empty() {
                return Ok(calls);
            }
            calls.push(self.call()?);
            if !self.eat(',') {
                if bracketed {
                    self.expect(']')?;
                }
                return Ok(calls);
            }
        }
    }

    fn call(&mut self) -> Result<ToolCall, String> {
        let name = self.identifier()?;
        self.expect('(')?;
        let mut arguments = Map::new();
        if !self.eat(')') {
            loop {
                let key = self.identifier()?;
                self.expect('=')?;
                let value = self.value()?;
                arguments.insert(key, value);
                if !self.eat(',') {
                    self.expect(')')?;
                    break;
                }
                if self.eat(')') {
                    break;
                }
            }
        }
        Ok(ToolCall { name, arguments })
    }

    fn identifier(&mut self) -> Result<String, String> {
        self.skip_space();
        let length = self
            .rest()
            .find(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.' || c == '-'))
            .unwrap_or(self.rest().len());
        if length == 0 {
            return Err(format!("expected a name at {:?}", self.rest()));
        }
        let name = self.rest()[..length].to_owned();
        self.pos += length;
        Ok(name)
    }

    fn value(&mut self) -> Result<Value, String> {
        self.skip_space();
        let rest = self.rest();
        if rest.starts_with('\'') || rest.starts_with('"') {
            return self.string().map(Value::String);
        }
        if self.eat('[') {
            let mut items = Vec::new();
            while !self.eat(']') {
                items.push(self.value()?);
                if !self.eat(',') {
                    self.expect(']')?;
                    break;
                }
            }
            return Ok(Value::Array(items));
        }
        if self.eat('{') {
            let mut map = Map::new();
            while !self.eat('}') {
                self.skip_space();
                let key = self.string()?;
                self.expect(':')?;
                map.insert(key, self.value()?);
                if !self.eat(',') {
                    self.expect('}')?;
                    break;
                }
            }
            return Ok(Value::Object(map));
        }
        let length = rest
            .find(|c: char| matches!(c, ',' | ')' | ']' | '}') || c.is_whitespace())
            .unwrap_or(rest.len());
        let word = &rest[..length];
        self.pos += length;
        match word {
            "True" | "true" => Ok(Value::Bool(true)),
            "False" | "false" => Ok(Value::Bool(false)),
            "None" | "null" => Ok(Value::Null),
            _ => serde_json::from_str::<serde_json::Number>(word)
                .map(Value::Number)
                .map_err(|_| format!("unknown value {word:?}")),
        }
    }

    /// Parse a string in single or double quotes with backslash escapes.
    fn string(&mut self) -> Result<String, String> {
        let mut chars = self.rest().char_indices();
        let Some((_, quote)) = chars.next().filter(|(_, c)| *c == '\'' || *c == '"') else {
            return Err(format!("expected a string at {:?}", self.rest()));
        };
        let mut out = String::new();
        while let Some((offset, c)) = chars.next() {
            match c {
                '\\' => match chars.next() {
                    Some((_, 'n')) => out.push('\n'),
                    Some((_, 'r')) => out.push('\r'),
                    Some((_, 't')) => out.push('\t'),
                    Some((_, other)) => out.push(other),
                    None => break,
                },
                c if c == quote => {
                    self.pos += offset + c.len_utf8();
                    return Ok(out);
                }
                c => out.push(c),
            }
        }
        Err("unterminated string".to_owned())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn python_calls_parse() {
        let calls = parse(
            "[get_weather(city='Paris, \\'FR\\'', days=3, metric=True), \
             search(query=\"cats\", filters={\"size\": [1, 2]}, page=None), now()]",
        )
        .unwrap();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments["city"], json!("Paris, 'FR'"));
        assert_eq!(calls[0].arguments["days"], json!(3));
        assert_eq!(calls[0].arguments["metric"], json!(true));
        assert_eq!(calls[1].arguments["filters"], json!({"size": [1, 2]}));
        assert_eq!(calls[1].arguments["page"], json!(null));
        assert!(calls[2].arguments.is_empty());
    }

    #[test]
    fn json_calls_parse() {
        let calls = parse(r#"[{"name": "f", "arguments": {"a": "b"}}]"#).unwrap();
        assert_eq!(calls[0].name, "f");
        assert_eq!(calls[0].arguments["a"], json!("b"));
    }

    #[test]
    fn formatted_calls_parse_back() {
        let call = ToolCall {
            name: "write".to_owned(),
            arguments: json!({"path": "a'b\nc", "n": 2.5, "ok": false, "list": [1, "x"]})
                .as_object()
                .unwrap()
                .clone(),
        };
        let text = format(std::slice::from_ref(&call));
        assert_eq!(parse(&text).unwrap(), vec![call]);
    }
}
