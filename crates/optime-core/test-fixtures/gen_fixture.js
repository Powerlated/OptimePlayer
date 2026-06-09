// Generates the golden parity fixture for optime-core's tests by rendering a demo SSEQ with the
// ORIGINAL legacy JS engine. The Rust engine must match this output sample-for-sample.
//
// Usage:  node gen_fixture.js <demo.sdat> [seconds]
//
// It auto-selects the first SSEQ that produces audible output, renders it, and writes:
//   golden.bin   - interleaved little-endian f32 stereo samples (L, R, L, R, ...)
//   golden.json  - { demo, sseqId, sseqName, sampleRate, seconds, frames }
//
// The render loop here is intentionally identical to Controller::next_sample in the Rust port:
// accumulate the DS clock per output sample, tick the controller every 64*2728*SAMPLE_RATE
// cycles, then advance all 16 synthesizers and sum them.

const fs = require("fs");
const path = require("path");
const vm = require("vm");

// Minimal browser-global stubs so the engine loads under Node (the synthesis path never calls
// into these, matching how legacy-js/video-exporter.js drives the engine headlessly).
global.window = {};
global.document = { addEventListener() {} };
global.alert = () => {};
global.AudioBuffer = function () {};

const enginePath = path.resolve(__dirname, "../../../legacy-js/OptimePlayer/OptimePlayer.js");
let src = fs.readFileSync(enginePath, "utf8");
// Top-level `class`/`let` declarations under runInThisContext are lexically scoped to the
// script, so explicitly export what we need onto globalThis.
src += "\n;globalThis.__optime = { Sdat, Controller };";
vm.runInThisContext(src);
const { Sdat, Controller } = globalThis.__optime;

// Silence the engine's verbose logging; keep our own messages on stderr.
console.log = () => {};

const SAMPLE_RATE = 32768;
const demoPath = process.argv[2] || path.resolve(__dirname, "../../../demos/super-mario-64-ds.sdat");
const seconds = Number(process.argv[3] || 4);

const romData = new Uint8Array(fs.readFileSync(demoPath));
const sdats = Sdat.loadAllFromDataView(new DataView(romData.buffer));
if (sdats.length === 0) {
    console.error("No SDAT found in " + demoPath);
    process.exit(1);
}
const sdat = sdats[0];

// Render `frames` samples of the given sseq id, returning a Float32 interleaved buffer plus the
// peak absolute amplitude seen (for the audible-output scan).
function render(id, frames) {
    const controller = new Controller(SAMPLE_RATE, sdat, id);
    const out = Buffer.alloc(frames * 2 * 4);
    let timer = 0;
    let off = 0;
    let peak = 0;
    for (let s = 0; s < frames; s++) {
        timer += 33513982;
        while (timer >= 64 * 2728 * SAMPLE_RATE) {
            timer -= 64 * 2728 * SAMPLE_RATE;
            controller.tick();
        }
        let valL = 0;
        let valR = 0;
        for (let i = 0; i < 16; i++) {
            const syn = controller.synthesizers[i];
            syn.nextSample();
            valL += syn.valL;
            valR += syn.valR;
        }
        out.writeFloatLE(valL, off);
        off += 4;
        out.writeFloatLE(valR, off);
        off += 4;
        peak = Math.max(peak, Math.abs(valL), Math.abs(valR));
    }
    return { out, peak };
}

// Find the first sseq that makes audible sound within its first half second.
let chosenId = null;
for (const id of sdat.sseqList) {
    try {
        const { peak } = render(id, Math.floor(SAMPLE_RATE * 0.5));
        if (peak > 0.05) {
            chosenId = id;
            break;
        }
    } catch (e) {
        // Skip sequences the engine can't build (missing bank/sample, etc.).
    }
}
if (chosenId === null) {
    console.error("Could not find an audible SSEQ in " + demoPath);
    process.exit(1);
}

const frames = Math.floor(SAMPLE_RATE * seconds);
const { out } = render(chosenId, frames);

const meta = {
    demo: path.basename(demoPath),
    sseqId: chosenId,
    sseqName: sdat.sseqIdNameDict.get(chosenId) || null,
    sampleRate: SAMPLE_RATE,
    seconds,
    frames,
};

fs.writeFileSync(path.resolve(__dirname, "golden.bin"), out);
fs.writeFileSync(path.resolve(__dirname, "golden.json"), JSON.stringify(meta, null, 2) + "\n");
console.error("Wrote golden fixture:", JSON.stringify(meta));
