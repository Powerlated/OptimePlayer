Your goal is to implement support for GBA games in an idiomatic, rusty manner. Whenever you can, use Rust idioms, make invalid state representable, and use enums.

Ensure the code is split up into files in a way where it is very easy for humans to see the data flow.

e.g. There is a devices folder than holds folders named Nintendo DS, GBA, etc. Inside each device folder, there is a set of structs (collectively deemed a device core) that generates a standardized set of messages that can be sent into the SynthController for synthesis. 

Use this decompiled version of the Pokemon Emerald source code as a reference for the ROM format an game engine. Clone it down: https://github.com/pret/pokeemerald

Options exist for the user to select between:
 * Crunchy. Just like the DS player's crunchy option.
   * In crunchy mode, there should be two sliders:
     * (1) Cutoff frequency of filter for PSGs
     * (2) Cutoff frequency of filter for DirectSound samplers
   * A checkbox where the user can choose to preserve or smooth out the pops and clicks from the PSGs abruptly turning on and off.
 * GBA Authentic. The sound coming out of this mode should be indistinguishable from real hardware output while playing Pokemon Emerald. This includes:
   * Linear interpolation takes DirectSound samples from their instrument sample rate to the mixer sample rate of 13379 Hz.
   * The 13379 Hz mixer sample rate is converted to the hardware output sample rate of 32768 Hz by nearest neighbor upsampling.
   * The 32768 Hz is taken to the OptimePlayer output sample rate by proper sample rate conversion.
   * PSGs are nearest neighbor sampled at 32768 Hz, the hardware output sample rate.
   * There should be sliders:
     * (1) Cutoff frequency of filter
 * GBA Crunchy Authentic. Same as authentic, but the 13379 Hz -> 32768 Hz conversion is done by a bandlimited zero-order hold.
  
Only sliders for the currently selected option should be visible.

GBA and NDS should have independent settings in case users want to listen to NDS in Crunchy and GBA in Authentic, for example.

The GBA device core and the SynthController must agree on pan laws. 

The amount of sinc taps in every resampler must be configurable.

The stereo expander (controlled by the "Stereo Separation" checkbox) must have an option for the user to choose how the player smooths out pops and clicks from delay line length changes:
 * No smoothing
 * Disallow delay line change in the middle of a note playing

The application should have a button for exporting the audio data of the GBA ROM so that the audio data can be shipped in demos/ without bundling any other data (e.g. code, sprites, etc.) from the ROM.  

When a ROM is loaded, only valid songs (i.e. clicking the song in the library will play it) should be added to the library view. 

You have the authority to refactor any code in any way you see fit to achieve maximum rustiness. 