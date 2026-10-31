# Architecture rules

Core is domain-neutral. Apps are plugins in their own repos. A change that
breaks one of these rules needs a design doc.

1. Core has no domain knowledge. No project, repository, review system, or
   tracker names in core code or prompts.
2. The bus is transport only. A payload is opaque JSON. Its schema belongs to
   the app that publishes it.
3. A plugin reaches core only through named seams: subscribers, message
   handlers, local tools, follow-ups. Core imports no plugin by name.
4. Message kind is set by the transport, never from content. A model cannot
   write a human message.
5. Three state homes: SQLite for state whose loss changes behaviour, the bus
   for messages, the artifact log for renderings. No fourth.
