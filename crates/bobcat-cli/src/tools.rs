//! Tool calls in the forms LFM2 and Qwen write them.
//!
//! LFM2 writes tool calls between its tool call tokens as a Python list of calls, as in
//! `[get_weather(city='Paris', days=3)]`. Keyword arguments hold Python literals: quoted
//! strings, numbers, `True`, `False`, `None`, and lists and dicts. When a system prompt asks for
//! JSON, the model writes a JSON list of `{"name", "arguments"}` objects in place of the
//! Python list, so the parser accepts both. Qwen3.5 writes each call between its own tool call
//! tags as a `<function=NAME>` block of `<parameter=NAME>` blocks, one per argument, with each
//! value on its own lines.

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
    if text.starts_with("<function=") {
        return Ok(vec![tagged_call(text)?]);
    }
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

/// Return the call in `text`, a `<function=NAME>` block as Qwen3.5 writes it.
///
/// A parameter's value is JSON when it parses as an object, a list, a number, or a JSON literal,
/// and Python's `True`, `False`, and `None` become their JSON values. Any other value is a string,
/// as Qwen3.5's chat template writes strings without quotes.
fn tagged_call(text: &str) -> Result<ToolCall, String> {
    let rest = text
        .strip_prefix("<function=")
        .ok_or_else(|| format!("expected <function= at {text:?}"))?;
    let (name, mut rest) = rest
        .split_once('>')
        .ok_or_else(|| format!("the function tag never closes in {text:?}"))?;
    let mut arguments = Map::new();
    loop {
        rest = rest.trim_start();
        if rest.starts_with("</function>") {
            return Ok(ToolCall {
                name: name.trim().to_owned(),
                arguments,
            });
        }
        let after = rest
            .strip_prefix("<parameter=")
            .ok_or_else(|| format!("expected <parameter= or </function> at {rest:?}"))?;
        let (key, after) = after
            .split_once('>')
            .ok_or_else(|| format!("the parameter tag never closes in {after:?}"))?;
        let (value, after) = after
            .split_once("</parameter>")
            .ok_or_else(|| format!("the parameter {key} never closes"))?;
        // The template writes each value on lines of its own.
        let value = value.strip_prefix('\n').unwrap_or(value);
        let value = value.strip_suffix('\n').unwrap_or(value);
        arguments.insert(key.trim().to_owned(), tagged_value(value));
        rest = after;
    }
}

/// Return the JSON value a parameter's text stands for.
fn tagged_value(text: &str) -> Value {
    match text.trim() {
        "True" => return Value::Bool(true),
        "False" => return Value::Bool(false),
        "None" => return Value::Null,
        _ => {}
    }
    match serde_json::from_str::<Value>(text) {
        Ok(Value::String(_)) | Err(_) => Value::String(text.to_owned()),
        Ok(value) => value,
    }
}

/// Return `calls` in the form Qwen3.5 writes them, each between `open` and `close`, for replaying
/// earlier calls in a conversation.
pub(crate) fn format_tagged(calls: &[ToolCall], open: &str, close: &str) -> String {
    calls
        .iter()
        .map(|call| {
            let mut text = format!("{open}\n<function={}>\n", call.name);
            for (key, value) in &call.arguments {
                let value = match value {
                    Value::String(text) => text.clone(),
                    Value::Bool(true) => "True".to_owned(),
                    Value::Bool(false) => "False".to_owned(),
                    Value::Null => "None".to_owned(),
                    Value::Number(_) | Value::Array(_) | Value::Object(_) => value.to_string(),
                };
                for piece in ["<parameter=", key, ">\n", &value, "\n</parameter>\n"] {
                    text.push_str(piece);
                }
            }
            text + "</function>\n" + close
        })
        .collect::<Vec<_>>()
        .join("\n")
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
    fn tagged_calls_parse_back() {
        let call = ToolCall {
            name: "search".to_owned(),
            arguments: json!({"query": "two\nlines", "n": 3, "exact": true, "tags": ["a"]})
                .as_object()
                .unwrap()
                .clone(),
        };
        let text = format_tagged(std::slice::from_ref(&call), "<tool_call>", "</tool_call>");
        let inner = text
            .strip_prefix("<tool_call>")
            .and_then(|text| text.strip_suffix("</tool_call>"))
            .unwrap();
        assert_eq!(parse(inner).unwrap(), vec![call]);
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
