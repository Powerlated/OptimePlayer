Your goal is to implement support for GBA games in an idiomatic, rusty manner. Whenever you can, use Rust idioms, make invalid state representable, and use enums.

Ensure the code is split up into files in a way where it is very easy for humans to see the data flow.

e.g. There is a devices folder than holds folders named Nintendo DS, GBA, etc. Inside each device folder, there is a set of structs that generates a standardized set of messages that can be sent into the SynthController (rename it from Controller) for synthesis. 

Use this decompiled version of the Pokemon Emerald source code as a reference for the ROM format an game engine. Clone it down: https://github.com/pret/pokeemerald

Before you begin, make sure CLAUDE.md is as accurate as possible to help you with architecture.