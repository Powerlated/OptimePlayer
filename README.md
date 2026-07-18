# Optime Player

This is a web and desktop application for playing music from various game console ROMs.

Currently, most games from these consoles are supported:
 * Nintendo DS
 * Game Boy Advance

Originally, the codebase was ~5000 lines of HTML+CSS+JS. It has since been translated to Rust with LLM assistance. From there on out, new features have been added, including GBA support, with heavy LLM assistance along the way. 

The eventual goal is for this project to feature a MIDI exporter, combined with a VST that leverages the Optime Player synthesizer core, that allows any artist in any DAW to create songs (namely High Quality Video Game Rips) that sound faithful to original console hardware. 

Thanks a lot to the https://github.com/pret/pokediamond and https://github.com/pret/pokeemerald projects for providing reverse engineered source code of the software side of the DS and GBA sound systems, respectively.

## Credits

Song-title data for the in-app library:

 * **Pokémon Emerald** — track titles follow the official *Pokémon Ruby, Pokémon Sapphire & Pokémon Emerald: Super Music Collection*, mapped to song ids via [pret/pokeemerald](https://github.com/pret/pokeemerald).
 * **Mother 3** — English Sound Player track names are from the [MOTHER 3 Fan Translation](https://mother3.fobby.net/) by the MOTHER 3 Fan Translation Team (Tomato/Jeffman et al.), paired with each track's in-game song id from the Sound Player's slot table in the ROM.

Vendored third-party code:

 * **[Vue.js](https://vuejs.org/)** (MIT, © Yuxi "Evan" You and Vue contributors) — the global build is vendored verbatim at `ml/dashboard/vendor/vue.global.prod.js` so the ML training dashboard serves itself with no build step and no CDN fetch.