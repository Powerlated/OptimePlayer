# Optime Player

This is a web application for playing music from Nintendo DS ROMs. 

Originally, the codebase was ~5000 lines of HTML+CSS+JS. It has since been translated to Rust with LLM assistance. From there on out, new features have been added, including GBA support, with heavy LLM assistance along the way. 

The eventual goal is for this project to feature a MIDI exporter, combined with a VST that leverages the Optime Player synthesizer core, that allows any artist in any DAW to create music (namely High Quality Video Game Rips) that sounds faithful to original console hardware. 

Thanks a lot to the https://github.com/pret/pokediamond and https://github.com/pret/pokeemerald projects for providing reverse engineered source code of the software side of the DS and GBA sound systems, respectively.