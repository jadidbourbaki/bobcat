//! A conversation in the terminal: the messages so far on top of an [`Engine`].

use std::sync::atomic::AtomicBool;

use crate::Error;
use crate::engine::{Engine, Event, Message, Reply, Request, Role};
use crate::sampler::Sampler;

/// A conversation with a model on the Metal GPU.
pub(crate) struct Conversation<'a> {
    engine: Engine<'a>,
    /// Every message so far, starting with the system message when there is one.
    messages: Vec<Message>,
    sampler: Sampler,
    max_tokens: usize,
}

impl<'a> Conversation<'a> {
    /// Start a conversation on `engine`, opened by the system message `system` when given, whose
    /// replies `sampler` chooses and hold at most `max_tokens` tokens.
    pub(crate) fn new(
        engine: Engine<'a>,
        system: Option<&str>,
        sampler: Sampler,
        max_tokens: usize,
    ) -> Self {
        let messages = system
            .map(|text| Message::new(Role::System, text))
            .into_iter()
            .collect();
        Self {
            engine,
            messages,
            sampler,
            max_tokens,
        }
    }

    /// Forget every message except the system message.
    pub(crate) fn clear(&mut self) {
        self.messages.retain(|message| message.role == Role::System);
    }

    /// Replace the system message with `text`, or remove it when `text` is empty.
    pub(crate) fn set_system(&mut self, text: &str) {
        self.messages.retain(|message| message.role != Role::System);
        if !text.is_empty() {
            self.messages.insert(0, Message::new(Role::System, text));
        }
    }

    /// Reply to the user message `text`, passing each event to `emit` as it decodes.
    ///
    /// The reply ends early when `cancel` becomes true. A failed reply leaves the messages as
    /// they were before `text`.
    pub(crate) fn reply(
        &mut self,
        text: &str,
        cancel: &AtomicBool,
        emit: impl FnMut(Event<'_>) -> Result<(), Error>,
    ) -> Result<Reply, Error> {
        self.messages.push(Message::new(Role::User, text));
        let mut request = Request {
            messages: &self.messages,
            tools: &[],
            sampler: &mut self.sampler,
            max_tokens: self.max_tokens,
            stop: &[],
        };
        let result = self.engine.generate(&mut request, cancel, emit);
        match &result {
            Ok(reply) => self
                .messages
                .push(Message::new(Role::Assistant, reply.answer.clone())),
            Err(_) => {
                self.messages.pop();
            }
        }
        result
    }
}
