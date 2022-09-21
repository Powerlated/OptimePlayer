// @ts-check

/** GLOBALS GO HERE **/
let debug = false;

let enableStereoSeparation = false;
let enableForceStereoSeparation = false;
let enableAntiAliasing = true;
let enableSoundgoodizer = true;
let elementarySchoolBandMode = false;

/** @type {ControllerBridge | null} */
let currentBridge = null;
/** @type {FsVisControllerBridge | null} */
let currentFsVisBridge = null;
/** @type {string | null} */
let currentlyPlayingName = null;
/** @type {Sdat | null} */
let currentlyPlayingSdat = null;
/** @type {number} */
let currentlyPlayingId = 0;
/** @type {AudioPlayer | null} */
let currentPlayer = null;

/** @type {boolean[]} */
let trackEnables = new Array(16).fill(true);

let synthtestSample;
let synthtest2Sample;


function polyBlep(t, dt) {
    // 0 <= t < 1
    if (t < dt) {
        t /= dt;
        // 2 * (t - t^2/2 - 0.5)
        return t + t - t * t - 1.0;
    }
    // -1 < t < 0
    else if (t > 1.0 - dt) {
        t = (t - 1.0) / dt;
        // 2 * (t^2/2 + t + 0.5)
        return t * t + t + t + 1.0;
    }
    // 0 otherwise
    else {
        return 0.0;
    }
}

function polyBlepTest(t, dt) {
    // 0 <= t < 1
    if (t < dt) {
        return 1;
    }
    // -1 < t < 0
    else if (t > 1.0 - dt) {
        return -1;
    }
    // 0 otherwise
    else {
        return 0.0;
    }
}


class SoundgoodizerFilterChannel {
            // Each biquad filter has a slope of 12db/oct so 2 biquads chained gets us 24db/oct
            /** @type {BiQuadFilter[]} */ lowFilters = new Array(2);
            /** @type {BiQuadFilter[]} */ midFilters = new Array(4);
            /** @type {BiQuadFilter[]} */ highFilters = new Array(2);

    outLow = 0;
    outMid = 0;
    outHigh = 0;

    dbPerOct24 = false;

    /**
     * @param {boolean} dbPerOct24
     * @param {number} sampleRate
     * @param {number} lowHz
     * @param {number} highHz
     **/
    constructor(dbPerOct24, sampleRate, lowHz, highHz) {
        this.dbPerOct24 = dbPerOct24;

        // q = 1/sqrt(2) maximally flat "butterworth" filter
        let q = 1 / Math.SQRT2;
        this.lowFilters[0] = BiQuadFilter.lowPassFilter(sampleRate, lowHz, q);
        this.lowFilters[1] = BiQuadFilter.lowPassFilter(sampleRate, lowHz, q);

        this.midFilters[0] = BiQuadFilter.highPassFilter(sampleRate, lowHz, q);
        this.midFilters[1] = BiQuadFilter.lowPassFilter(sampleRate, highHz, q);
        this.midFilters[2] = BiQuadFilter.highPassFilter(sampleRate, lowHz, q);
        this.midFilters[3] = BiQuadFilter.lowPassFilter(sampleRate, highHz, q);

        this.highFilters[0] = BiQuadFilter.highPassFilter(sampleRate, highHz, q);
        this.highFilters[1] = BiQuadFilter.highPassFilter(sampleRate, highHz, q);
    }

    /**
     * @param {boolean} dbPerOct24
     * @param {number} sampleRate
     * @param {number} lowHz
     * @param {number} highHz
     * */
    changeFilterParams(dbPerOct24, sampleRate, lowHz, highHz) {
        this.dbPerOct24 = dbPerOct24;

        let q = 1 / Math.SQRT2;
        this.lowFilters[0].setLowPassFilter(sampleRate, lowHz, q);
        this.lowFilters[1].setLowPassFilter(sampleRate, lowHz, q);

        this.midFilters[0].setHighPassFilter(sampleRate, lowHz, q);
        this.midFilters[1].setLowPassFilter(sampleRate, highHz, q);
        this.midFilters[2].setHighPassFilter(sampleRate, lowHz, q);
        this.midFilters[3].setLowPassFilter(sampleRate, highHz, q);

        this.highFilters[0].setHighPassFilter(sampleRate, highHz, q);
        this.highFilters[1].setHighPassFilter(sampleRate, highHz, q);
    }

    /**
     * @param {number} inVal
     * */
    process(inVal) {
        this.outLow = inVal;
        this.outMid = inVal;
        this.outHigh = inVal;

        for (let i = 0; i < (this.dbPerOct24 ? 2 : 1); i++) {
            this.outLow = this.lowFilters[i].transform(this.outLow);
        }

        for (let i = 0; i < (this.dbPerOct24 ? 4 : 2); i++) {
            this.outMid = this.midFilters[i].transform(this.outMid);
        }

        for (let i = 0; i < (this.dbPerOct24 ? 2 : 1); i++) {
            this.outHigh = this.highFilters[i].transform(this.outHigh);
        }
    }
}

// The entire point of using a filter to downsample is to antialias with subsample resolution in the output signal
// We need to decide how precise our filter is
const KERNEL_RESOLUTION = 1024;
class BlipBuf {
    // Lanzcos kernel
    /** @type {Float64Array} */ kernel;
    kernelSize = 0;

    /** @type {Float64Array} */ channelVals;
    /** @type {Float64Array} */ channelSample;
    /** @type {Float64Array} */ channelRealSample;

    // This is a buffer of differences we are going to write bandlimited impulses to
     /** @type {Float64Array} */ buf;
    bufPos = 0;
    bufSize = 0;

    currentVal = 0;

    currentSampleInPos = 0;
    currentSampleOutPos = 0;

    /**
     * @param {number} kernelSize 
     * @param {boolean} normalize 
     * @param {number} channels 
     * @param {number} filterRatio 
     */
    constructor(kernelSize, normalize, channels, filterRatio) {
        this.channelVals = new Float64Array(channels);
        this.channelSample = new Float64Array(channels);
        this.channelRealSample = new Float64Array(channels);

        this.bufSize = 32768;
        this.buf = new Float64Array(this.bufSize);

        this.setKernelSize(kernelSize, normalize, filterRatio);
    }

    /**
     * @param {number} kernelSize 
     * @param {boolean} normalize 
     * @param {number} filterRatio 
     */
    setKernelSize(kernelSize, normalize, filterRatio) {
        this.kernel = new Float64Array(kernelSize * KERNEL_RESOLUTION);
        this.kernelSize = kernelSize;

        if ((kernelSize & (kernelSize - 1)) != 0) {
            throw "kernelSize not power of 2:" + kernelSize;
        }

        if (filterRatio <= 0 || filterRatio > Math.PI) {
            throw "invalid filterRatio, outside of (0, pi]";
        }

        // Generate the normalized Lanzcos kernel
        // Derived from Wikipedia https://en.wikipedia.org/wiki/Lanczos_resampling
        for (let i = 0; i < KERNEL_RESOLUTION; i++) {
            let sum = 0;
            for (let j = 0; j < kernelSize; j++) {
                let x = j - kernelSize / 2;
                // Shift X coordinate right for subsample accuracy
                // We now have the X coordinates for an impulse bandlimited at the sample rate
                x += (KERNEL_RESOLUTION - i - 1) / KERNEL_RESOLUTION;
                // filterRatio determines the highest frequency that will be generated,
                // as a portion of the Nyquist frequency. e.g. at a sample rate of 48000 hz, the Nyquist frequency
                // will be at 24000 hz, so a filterRatio of 0.5 would set the highest frequency
                // generated at 12000 hz.
                x *= filterRatio * Math.PI;

                // Get the sinc, which represents a bandlimited impulse
                let sinc = Math.sin(x) / x;
                // A sinc function's domain is infinte, meaning 
                // convolving a signal with a true sinc function would take an infinite amount of time
                // To avoid creating a filter with infinite latency, we have to decide when to cut off
                // our sinc function. We can window (i.e. multiply) our true sinc function with a
                // horizontally stretched sinc function to create a windowed sinc function of our desired width. 
                let lanzcosWindow = Math.sin(x / kernelSize) / (x / kernelSize);

                // A hole exists in the sinc function at zero, special case it
                if (x == 0) {
                    this.kernel[i * kernelSize + j] = 1;
                }
                else {
                    // Apply our window here
                    this.kernel[i * kernelSize + j] = sinc * lanzcosWindow;
                }

                sum += this.kernel[i * kernelSize + j];
            }

            if (normalize) {
                for (let j = 0; j < kernelSize; j++) {
                    this.kernel[i * kernelSize + j] /= sum;
                }
            }
        }

        // this.reset();
    }

    reset() {
        // Flush out the difference buffer
        this.bufPos = 0;
        this.currentVal = 0;
        for (let i = 0; i < this.bufSize; i++) {
            this.buf[i] = 0;
        }
    }

    // Sample is in terms of out samples
    setValue(channel, sample, val, useSinc) {
        if (sample > this.channelSample[channel]) {
            this.channelSample[channel] = sample;
        } else {
            // TODO: too lazy to fix anything regarding this so I'll just sweep it under the rug for now
            // console.warn(`Channel ${channel}: Tried to set amplitude backward in time from ${this.channelSample[channel]} to ${sample}`);
        }

        if (val != this.channelVals[channel]) {
            let diff = val - this.channelVals[channel];

            if (useSinc) {
                let subsamplePos = Math.floor((sample % 1) * KERNEL_RESOLUTION);

                // Add our bandlimited impulse to the difference buffer
                let kBufPos = (this.bufPos + Math.floor(sample) - this.currentSampleOutPos) % this.bufSize;
                for (let i = 0; i < this.kernelSize; i++) {
                    this.buf[kBufPos] += this.kernel[this.kernelSize * subsamplePos + i] * diff;
                    if (++kBufPos >= this.bufSize) kBufPos = 0;
                }
            } else {
                let kBufPos = (this.bufPos + Math.floor(sample + this.kernelSize) - this.currentSampleOutPos) % this.bufSize;
                this.buf[kBufPos] += diff;
            }
        }

        this.channelVals[channel] = val;
    }

    readOutSample() {
        // Integrate the difference buffer
        this.currentVal += this.buf[this.bufPos];
        this.buf[this.bufPos] = 0;
        if (++this.bufPos >= this.bufSize) this.bufPos = 0;
        this.currentSampleOutPos++;
        return this.currentVal;
    }
}

class LowPass96DbPerOct {
    constructor(sampleRate, cutoffFrequency) {
        this.f0 = BiQuadFilter.lowPassFilter(sampleRate, cutoffFrequency, Math.SQRT1_2);
        this.f1 = BiQuadFilter.lowPassFilter(sampleRate, cutoffFrequency, Math.SQRT1_2);
        this.f2 = BiQuadFilter.lowPassFilter(sampleRate, cutoffFrequency, Math.SQRT1_2);
        this.f3 = BiQuadFilter.lowPassFilter(sampleRate, cutoffFrequency, Math.SQRT1_2);
    }

    transform(inSample) {
        inSample = this.f0.transform(inSample);
        inSample = this.f1.transform(inSample);
        inSample = this.f2.transform(inSample);
        return this.f3.transform(inSample);
    }

    resetState() {
        this.f0.resetState();
        this.f1.resetState();
        this.f2.resetState();
        this.f3.resetState();
    }

    set(sampleRate, cutoffFrequency) {
        this.f0.setLowPassFilter(sampleRate, cutoffFrequency, Math.SQRT1_2)
        this.f1.setLowPassFilter(sampleRate, cutoffFrequency, Math.SQRT1_2)
        this.f2.setLowPassFilter(sampleRate, cutoffFrequency, Math.SQRT1_2)
        this.f3.setLowPassFilter(sampleRate, cutoffFrequency, Math.SQRT1_2)
    }
}

// Based off NAudio's BiQuadFilter, look there for comments
class BiQuadFilter {
    // coefficients
    a0 = 0;
    a1 = 0;
    a2 = 0;
    a3 = 0;
    a4 = 0;

    // state
    x1 = 0;
    x2 = 0;
    y1 = 0;
    y2 = 0;

    resetState() {
        this.x1 = 0;
        this.x2 = 0;
        this.y1 = 0;
        this.y2 = 0;
    }

    transform(inSample) {
        // compute result
        let result = this.a0 * inSample + this.a1 * this.x1 + this.a2 * this.x2 - this.a3 * this.y1 - this.a4 * this.y2;
        if (isNaN(result)) throw "sdf ";
        if (result == Infinity) throw "infinity??"; 

        // shift x1 to x2, sample to x1 
        this.x2 = this.x1;
        this.x1 = inSample;

        // shift y1 to y2, result to y1 
        this.y2 = this.y1;
        this.y1 = result;

        return result;
    }
    setCoefficients(aa0, aa1, aa2, b0, b1, b2) {
        this.a0 = b0 / aa0;
        this.a1 = b1 / aa0;
        this.a2 = b2 / aa0;
        this.a3 = aa1 / aa0;
        this.a4 = aa2 / aa0;
    }

    setLowPassFilter(sampleRate, cutoffFrequency, q) {
        let w0 = 2 * Math.PI * cutoffFrequency / sampleRate;
        let cosw0 = Math.cos(w0);
        let alpha = Math.sin(w0) / (2 * q);

        let b0 = (1 - cosw0) / 2;
        let b1 = 1 - cosw0;
        let b2 = (1 - cosw0) / 2;
        let aa0 = 1 + alpha;
        let aa1 = -2 * cosw0;
        let aa2 = 1 - alpha;
        this.setCoefficients(aa0, aa1, aa2, b0, b1, b2);
    }

    setPeakingEq(sampleRate, centreFrequency, q, dbGain) {
        let w0 = 2 * Math.PI * centreFrequency / sampleRate;
        let cosw0 = Math.cos(w0);
        let sinw0 = Math.sin(w0);
        let alpha = sinw0 / (2 * q);
        let a = Math.pow(10, dbGain / 40);

        let b0 = 1 + alpha * a;
        let b1 = -2 * cosw0;
        let b2 = 1 - alpha * a;
        let aa0 = 1 + alpha / a;
        let aa1 = -2 * cosw0;
        let aa2 = 1 - alpha / a;
        this.setCoefficients(aa0, aa1, aa2, b0, b1, b2);
    }

    setHighPassFilter(sampleRate, cutoffFrequency, q) {
        let w0 = 2 * Math.PI * cutoffFrequency / sampleRate;
        let cosw0 = Math.cos(w0);
        let alpha = Math.sin(w0) / (2 * q);

        let b0 = (1 + cosw0) / 2;
        let b1 = -(1 + cosw0);
        let b2 = (1 + cosw0) / 2;
        let aa0 = 1 + alpha;
        let aa1 = -2 * cosw0;
        let aa2 = 1 - alpha;
        this.setCoefficients(aa0, aa1, aa2, b0, b1, b2);
    }

    static lowPassFilter(sampleRate, cutoffFrequency, q) {
        let filter = new BiQuadFilter();
        filter.setLowPassFilter(sampleRate, cutoffFrequency, q);
        return filter;
    }

    static highPassFilter(sampleRate, cutoffFrequency, q) {
        let filter = new BiQuadFilter();
        filter.setHighPassFilter(sampleRate, cutoffFrequency, q);
        return filter;
    }

    static bandPassFilterConstantSkirtGain(sampleRate, centreFrequency, q) {
        let w0 = 2 * Math.PI * centreFrequency / sampleRate;
        let cosw0 = Math.cos(w0);
        let sinw0 = Math.sin(w0);
        let alpha = sinw0 / (2 * q);

        let b0 = sinw0 / 2;
        let b1 = 0;
        let b2 = -sinw0 / 2;
        let a0 = 1 + alpha;
        let a1 = -2 * cosw0;
        let a2 = 1 - alpha;
        return new BiQuadFilter(a0, a1, a2, b0, b1, b2);
    }

    static bandPassFilterConstantPeakGain(sampleRate, centreFrequency, q) {
        let w0 = 2 * Math.PI * centreFrequency / sampleRate;
        let cosw0 = Math.cos(w0);
        let sinw0 = Math.sin(w0);
        let alpha = sinw0 / (2 * q);

        let b0 = alpha;
        let b1 = 0;
        let b2 = -alpha;
        let a0 = 1 + alpha;
        let a1 = -2 * cosw0;
        let a2 = 1 - alpha;
        return new BiQuadFilter(a0, a1, a2, b0, b1, b2);
    }

    static notchFilter(sampleRate, centreFrequency, q) {
        let w0 = 2 * Math.PI * centreFrequency / sampleRate;
        let cosw0 = Math.cos(w0);
        let sinw0 = Math.sin(w0);
        let alpha = sinw0 / (2 * q);

        let b0 = 1;
        let b1 = -2 * cosw0;
        let b2 = 1;
        let a0 = 1 + alpha;
        let a1 = -2 * cosw0;
        let a2 = 1 - alpha;
        return new BiQuadFilter(a0, a1, a2, b0, b1, b2);
    }

    static allPassFilter(sampleRate, centreFrequency, q) {
        let w0 = 2 * Math.PI * centreFrequency / sampleRate;
        let cosw0 = Math.cos(w0);
        let sinw0 = Math.sin(w0);
        let alpha = sinw0 / (2 * q);

        let b0 = 1 - alpha;
        let b1 = -2 * cosw0;
        let b2 = 1 + alpha;
        let a0 = 1 + alpha;
        let a1 = -2 * cosw0;
        let a2 = 1 - alpha;
        return new BiQuadFilter(a0, a1, a2, b0, b1, b2);
    }

    static peakingEQ(sampleRate, centreFrequency, q, dbGain) {
        let filter = new BiQuadFilter();
        filter.setPeakingEq(sampleRate, centreFrequency, q, dbGain);
        return filter;
    }

    static lowShelf(sampleRate, cutoffFrequency, shelfSlope, dbGain) {
        let w0 = 2 * Math.PI * cutoffFrequency / sampleRate;
        let cosw0 = Math.cos(w0);
        let sinw0 = Math.sin(w0);
        let a = Math.pow(10, dbGain / 40);
        let alpha = sinw0 / 2 * Math.sqrt((a + 1 / a) * (1 / shelfSlope - 1) + 2);
        let temp = 2 * Math.sqrt(a) * alpha;

        let b0 = a * ((a + 1) - (a - 1) * cosw0 + temp);
        let b1 = 2 * a * ((a - 1) - (a + 1) * cosw0);
        let b2 = a * ((a + 1) - (a - 1) * cosw0 - temp);
        let a0 = (a + 1) + (a - 1) * cosw0 + temp;
        let a1 = -2 * ((a - 1) + (a + 1) * cosw0);
        let a2 = (a + 1) + (a - 1) * cosw0 - temp;
        return new BiQuadFilter(a0, a1, a2, b0, b1, b2);
    }

    static highShelf(sampleRate, cutoffFrequency, shelfSlope, dbGain) {
        let w0 = 2 * Math.PI * cutoffFrequency / sampleRate;
        let cosw0 = Math.cos(w0);
        let sinw0 = Math.sin(w0);
        let a = Math.pow(10, dbGain / 40);
        let alpha = sinw0 / 2 * Math.sqrt((a + 1 / a) * (1 / shelfSlope - 1) + 2);
        let temp = 2 * Math.sqrt(a) * alpha;

        let b0 = a * ((a + 1) + (a - 1) * cosw0 + temp);
        let b1 = -2 * a * ((a - 1) + (a + 1) * cosw0);
        let b2 = a * ((a + 1) + (a - 1) * cosw0 - temp);
        let a0 = (a + 1) - (a - 1) * cosw0 + temp;
        let a1 = 2 * ((a - 1) - (a + 1) * cosw0);
        let a2 = (a + 1) - (a - 1) * cosw0 - temp;
        return new BiQuadFilter(a0, a1, a2, b0, b1, b2);
    }

    constructor(a0 = 0, a1 = 0, a2 = 0, b0 = 0, b1 = 0, b2 = 0) {
        this.setCoefficients(a0, a1, a2, b0, b1, b2);
    }
}

function downloadUint8Array(name, array) {
    let blob = new Blob([array], { type: "application/octet-stream" });
    let link = document.createElement('a');
    link.href = window.URL.createObjectURL(blob);
    link.download = name;
    link.click();
}

//@ts-check
class WavEncoder {
    constructor(sampleRate, bits) {
        this.sampleRate = sampleRate;
        this.bits = bits;

        if (bits % 8 != 0) {
            alert("WavDownloader.constructor: bits not multiple of 8:" + bits);
        }
    }

    recordBuffer = new Uint8ClampedArray(32);
    recordBufferAt = 0;
    addSamples(left, right, size) {
        for (let i = 0; i < size; i++) {
            this.addSample(left[i], right[i]);
        }
    }

    addSample(left, right) {
        if (this.recordBufferAt + 16 > this.recordBuffer.length) {
            const oldBuf = this.recordBuffer;
            this.recordBuffer = new Uint8ClampedArray(this.recordBufferAt * 2);
            this.recordBuffer.set(oldBuf);
        }

        switch (this.bits) {
            case 8:
                this.recordBuffer[this.recordBufferAt++] = clamp(Math.round(((left + 1) / 2) * 255), 0, 255);
                this.recordBuffer[this.recordBufferAt++] = clamp(Math.round(((right + 1) / 2) * 255), 0, 255);
                break;
            case 16:
                const out0_16bit = clamp(Math.round(left * 32767), -32768, 32767);
                const out1_16bit = clamp(Math.round(right * 32767), -32768, 32767);
                this.recordBuffer[this.recordBufferAt++] = out0_16bit & 0xFF;
                this.recordBuffer[this.recordBufferAt++] = (out0_16bit >> 8) & 0xFF;
                this.recordBuffer[this.recordBufferAt++] = out1_16bit & 0xFF;
                this.recordBuffer[this.recordBufferAt++] = (out1_16bit >> 8) & 0xFF;
                break;
        }

    }

    encode() {
        // Allocate exactly enough for a WAV header
        const wave = new Uint8Array(this.recordBufferAt + 44);

        // RIFF header
        wave[0] = 0x52;
        wave[1] = 0x49;
        wave[2] = 0x46;
        wave[3] = 0x46;

        const size = wave.length - 8;
        wave[4] = (size >> 0) & 0xFF;
        wave[5] = (size >> 8) & 0xFF;
        wave[6] = (size >> 16) & 0xFF;
        wave[7] = (size >> 24) & 0xFF;

        // WAVE
        wave[8] = 0x57;
        wave[9] = 0x41;
        wave[10] = 0x56;
        wave[11] = 0x45;

        // Subchunk1ID "fmt "
        wave[12] = 0x66;
        wave[13] = 0x6d;
        wave[14] = 0x74;
        wave[15] = 0x20;

        // Subchunk1Size
        wave[16] = 16;
        wave[17] = 0;
        wave[18] = 0;
        wave[19] = 0;

        // AudioFormat
        wave[20] = 1;
        wave[21] = 0;

        // 2 channels
        wave[22] = 2;
        wave[23] = 0;

        // Sample rate
        wave[24] = (this.sampleRate >> 0) & 0xFF;
        wave[25] = (this.sampleRate >> 8) & 0xFF;
        wave[26] = (this.sampleRate >> 16) & 0xFF;
        wave[27] = (this.sampleRate >> 24) & 0xFF;

        // ByteRate
        // SampleRate & NumChannels * BitsPerSample/8
        const byteRate = this.sampleRate * 2 * (this.bits / 8);
        wave[28] = (byteRate >> 0) & 0xFF;
        wave[29] = (byteRate >> 8) & 0xFF;
        wave[30] = (byteRate >> 16) & 0xFF;
        wave[31] = (byteRate >> 24) & 0xFF;

        // BlockAlign
        // NumChannels * BitsPerSample / 8
        const blockAlign = 2 * (this.bits / 8);
        wave[32] = (blockAlign >> 0) & 0xFF;
        wave[33] = (blockAlign >> 8) & 0xFF;

        // BitsPerSample
        wave[34] = this.bits;
        wave[35] = 0;

        // Subchunk2ID "data"
        wave[36] = 0x64;
        wave[37] = 0x61;
        wave[38] = 0x74;
        wave[39] = 0x61;

        // NumSamples * NumChannels * BitsPerSample/8
        const subchunk2Size = this.recordBufferAt * 2 * (this.bits / 8);
        wave[40] = (subchunk2Size >> 0) & 0xFF;
        wave[41] = (subchunk2Size >> 8) & 0xFF;
        wave[42] = (subchunk2Size >> 16) & 0xFF;
        wave[43] = (subchunk2Size >> 24) & 0xFF;

        for (let i = 0; i < this.recordBufferAt; i++) {
            wave[44 + i] = this.recordBuffer[i];
        }

        return wave;
    }
}

class AudioPlayer {
    bufferLength;
    sampleRate;
    needMoreSamples;

    bufferPool;
    bufferPoolAt = 0;

    safariHax = false;

    constructor(bufferLength, sampleRate, needMoreSamples) {
        if (!AudioBuffer.prototype.copyToChannel) this.safariHax = true;

        this.bufferLength = bufferLength;
        this.sampleRate = sampleRate;
        this.needMoreSamples = needMoreSamples;

        const AudioContext = window.AudioContext   // Normal browsers
            //@ts-ignore
            || window.webkitAudioContext; // Sigh... Safari

        this.ctx = new AudioContext({ sampleRate: sampleRate });

        this.bufferPool = this.genBufferPool(256, this.bufferLength);

        const fixAudioContext = () => {
            // Create empty buffer
            let buffer = this.ctx.createBuffer(1, 1, 22050);

            /** @type {any} */
            let source = this.ctx.createBufferSource();
            source.buffer = buffer;
            // Connect to output (speakers)
            source.connect(this.ctx.destination);
            // Play sound
            if (source.start) {
                source.start(0);
            } else if (source.play) {
                source.play(0);
            } else if (source.noteOn) {
                source.noteOn(0);
            }
        };
        // iOS 6-8
        document.addEventListener('touchstart', fixAudioContext);
        // iOS 9
        document.addEventListener('touchend', fixAudioContext);

        this.gain = this.ctx.createGain();
        this.gain.gain.value = 0.25;
        this.gain.connect(this.ctx.destination);
    }

    gain;

    /** @type {AudioContext} */
    ctx;
    sourcesPlaying = 0;

    genBufferPool(count, length) {
        let pool = new Array(count);
        for (let i = 0; i < count; i++) {
            pool[i] = this.ctx.createBuffer(2, length, this.sampleRate);
        }
        return pool;
    }

    queueAudio(bufferLeft, bufferRight) {
        let buffer = this.bufferPool[this.bufferPoolAt];
        this.bufferPoolAt++;
        this.bufferPoolAt &= 255;

        buffer.getChannelData(0).set(bufferLeft);
        buffer.getChannelData(1).set(bufferRight);

        let bufferSource = this.ctx.createBufferSource();

        bufferSource.onended = () => {
            this.sourcesPlaying--;
            if (this.sourcesPlaying < 3) {
                this.needMoreSamples();
            }
            if (this.sourcesPlaying < 2) {
                this.needMoreSamples();
            }
        };

        // if (this.audioSec <= this.ctx.currentTime + 0.05) {
        // Reset time if close to buffer underrun
        // this.audioSec = this.ctx.currentTime + 0.06;
        // }
        bufferSource.buffer = buffer;
        bufferSource.connect(this.gain);
        bufferSource.start(this.audioSec);

        this.sampleRate = this.sampleRate;
        this.audioSec += this.bufferLength / this.sampleRate;

        this.sourcesPlaying++;

        // prevent dropouts when starting synthesis
        if (this.sourcesPlaying < 6) {
            this.needMoreSamples();
        }
        if (this.sourcesPlaying < 5) {
            this.needMoreSamples();
        }
    }

    audioSec = 0;

    reset() {
        // 50 ms buffer
        this.audioSec = this.ctx.currentTime + 0.06;
        // console.log(`Latency in seconds: ${(LATENCY / this.sampleRate)}`)
    }
}

function read16LE(data, addr) {
    return data[addr] | (data[addr + 1] << 8);
}

function read32LE(data, addr) {
    return data[addr] | (data[addr + 1] << 8) | (data[addr + 2] << 16) | (data[addr + 3] << 24);
}

function pad(n, width, z) {
    z = z || '0';
    n = n + '';
    return n.length >= width ? n : new Array(width - n.length + 1).join(z) + n;
}

function hex(i, digits) {
    return `0x${pad(i.toString(16), digits, '0').toUpperCase()}`;
}

function hexN(i, digits) {
    return pad(i.toString(16), digits, '0').toUpperCase();
}

/** @template T */
class CircularBuffer {
    /** @param {number} size */
    constructor(size) {
        this.size = size;
        /** @type T[] */
        this.buffer = new Array(size);

        this.entries = 0;
        this.readPos = 0;
        this.writePos = 0;
    }

    /** @param {T} data */
    insert(data) {
        if (this.entries < this.size) {
            this.entries++;
            this.buffer[this.writePos++] = data;

            if (this.writePos >= this.size) {
                this.writePos = 0;
            }

            return true;
        }

        throw "CircularBuffer: overflow";

        return false;
    }

    /** @returns {T} */
    pop() {
        let data;
        if (this.entries > 0) {
            this.entries--;
            data = this.buffer[this.readPos++];

            if (this.readPos >= this.size) {
                this.readPos = 0;
            }
        }
        else {
            throw "CircularBuffer: underflow";
            data = null;
        }
        return data;
    }

    /** @returns {T} */
    peek(offset) {
        return this.buffer[(this.readPos + offset) % this.size];
    }

    reset() {
        this.entries = 0;
        this.readPos = 0;
        this.writePos = 0;
    }
}

class SseqInfo {
    constructor() {
        this.fileId = null;
        this.bank = null;
        this.volume = null;
        this.cpr = null; // what the hell does this mean?
        this.ppr = null; // what the hell does this mean?
        this.ply = null; // what the hell does this mean?
    }
}

class SsarInfo {
    constructor() {
        this.fileId = null;
    }
}

class BankInfo {
    constructor() {
        this.fileId = null;
        this.swarId = new Uint16Array(4);
    }
}

class SwarInfo {
    constructor() {
        this.fileId = null;
    }
}

class Sdat {
    constructor() {
        /** @type {Uint8Array[]} */
        this.fat = [];

        this.sseqList = [];

        /** @type {(SseqInfo | null)[]} */
        this.sseqInfos = [];
        this.sseqNameIdDict = [];
        this.sseqIdNameDict = [];

        this.sbnkNameIdDict = [];
        this.sbnkIdNameDict = [];

        /** @type {(SsarInfo | null)[]} */
        this.ssarInfos = [];

        /** @type {(BankInfo | null)[]} */
        this.sbnkInfos = [];

        /** @type {(SwarInfo | null)[]} */
        this.swarInfos = [];

        /** @type {Bank[]} */
        this.banks = new Array(128);

        /** @type {Object.<number, Sample[]>} */
        this.sampleArchives = {};
    }
}

class Message {
    constructor(fromKeyboard, channel, type, param0, param1, param2) {
        this.fromKeyboard = fromKeyboard;
        this.trackNum = channel;
        this.type = type;
        this.param0 = param0;
        this.param1 = param1;
        this.param2 = param2;
        this.timestamp = 0;
    }
}

const MessageType = {
    PlayNote: 0, // P0: MIDI Note P1: Velocity P2: Duration
    InstrumentChange: 1, // P0: Bank P1: Program
    Jump: 2,
    TrackEnded: 3,
    VolumeChange: 4, // P0: Volume
    PanChange: 5, // P0: Pan (0-127)
    PitchBend: 6,
};

class Sample {
    /**
    * @param {Float32Array} data 
    * @param {number} frequency
    * @param {number} sampleRate
    * @param {boolean} looping
    * @param {number} loopPoint
    *
    */
    constructor(data, frequency, sampleRate, looping, loopPoint) {
        this.data = data;
        this.frequency = frequency;
        this.sampleRate = sampleRate;
        this.looping = looping;
        this.loopPoint = loopPoint;

        this.sampleLength = 0;

        this.enableFilter = true;
    }
}

const InstrumentType = {
    SingleSample: 0x1,
    PsgPulse: 0x2,
    PsgNoise: 0x3,

    Drumset: 0x10,
    MultiSample: 0x11
};

class InstrumentRecord {
    // fRecord = 0x1 - Single-Region Instrument
    // fRecord = 0x2 - PSG Pulse
    // fRecord = 0x3 - PSG Noise

    // fRecord = 0x10 - Drumset
    // fRecord = 0x11 - Multi-Region Instrument

    constructor() {
        this.fRecord = 0;

        this.lowerNote = 0;
        this.upperNote = 0;

        this.regionEnd = new Uint8Array(8);

        this.swavInfoId = [];
        this.swarInfoId = [];
        this.noteNumber = [];
        this.attack = [];
        this.attackCoefficient = [];
        this.decay = [];
        this.decayCoefficient = [];
        this.sustain = [];
        this.sustainLevel = [];
        this.release = [];
        this.releaseCoefficient = [];
        this.pan = [];
    }

    /** @returns {number} */
    resolveEntryIndex(note) {
        switch (this.fRecord) {
            case InstrumentType.SingleSample:
            case InstrumentType.PsgPulse:
            case InstrumentType.PsgNoise:
                return 0;

            case InstrumentType.Drumset:
                if (note < this.lowerNote || note > this.upperNote) {
                    console.warn(`resolveEntryIndex: drumset note out of range (${this.lowerNote}-${this.upperNote} inclusive): ${note}`);
                }
                return note - this.lowerNote;

            case InstrumentType.MultiSample:
                for (let i = 0; i < 8; i++) {
                    if (note <= this.regionEnd[i]) return i;
                }
                return 7;
        }

        return 0;
    }
}

// SBNK
class Bank {
    constructor() {
        /** @type {InstrumentRecord[]} */
        this.instruments = [];
    }
}

let filterSamples = true;

class SampleInstrument {
    /**
    * @param {number} sampleRate 
    * @param {Sample} sample
    */
    constructor(sampleRate, sample) {
        this.sampleRate = sampleRate;
        this.nyquist = sampleRate / 2;

        this.secondsPerSample = 1 / sampleRate;
        /** @type {Sample} */
        this.sample = sample;

        // sampleFrequency is the sample's tone frequency when played at sampleSampleRate
        this.val = 0;
        this.frequency = 440;
        this.volume = 1;

        this.playing = false;

        this.pan = 0.5;

        this.startTime = 0;

        this.midiNote = 0;

        this.sampleT = 0;

        this.finetune = 0;
        this.finetuneLfo = 0;

        this.freqRatio = 0;

        // We use a sloping filter rather than a brick wall to allow some of the image harmonics through
        this.filter = new LowPass96DbPerOct(this.sampleRate, 16384);

        // trackEnables.fill(false);
        // trackEnables[7] = true;

        Object.seal(this);
    }

    static {
        this.kernelSize = 4;
        this.setKernelSize(this.kernelSize, true, 1);
    }

    /**
     * @param {number} kernelSize 
     * @param {boolean} normalize 
     * @param {number} filterRatio 
     */
    static setKernelSize(kernelSize, normalize, filterRatio) {
        this.kernel = new Float64Array(kernelSize * KERNEL_RESOLUTION);
        this.kernelSize = kernelSize;

        if ((kernelSize & (kernelSize - 1)) != 0) {
            throw "kernelSize not power of 2:" + kernelSize;
        }

        if (filterRatio <= 0 || filterRatio > Math.PI) {
            throw "invalid filterRatio, outside of (0, pi]";
        }

        // Generate the normalized Lanzcos kernel
        // Derived from Wikipedia https://en.wikipedia.org/wiki/Lanczos_resampling
        for (let i = 0; i < KERNEL_RESOLUTION; i++) {
            let sum = 0;
            for (let j = 0; j < kernelSize; j++) {
                let x = j - kernelSize / 2;
                // Shift X coordinate right for subsample accuracy
                // We now have the X coordinates for an impulse bandlimited at the sample rate
                x += (KERNEL_RESOLUTION - i - 1) / KERNEL_RESOLUTION;
                // filterRatio determines the highest frequency that will be generated,
                // as a portion of the Nyquist frequency. e.g. at a sample rate of 48000 hz, the Nyquist frequency
                // will be at 24000 hz, so a filterRatio of 0.5 would set the highest frequency
                // generated at 12000 hz.
                x *= filterRatio * Math.PI;

                // Get the sinc, which represents a bandlimited impulse
                let sinc = Math.sin(x) / x;
                // A sinc function's domain is infinte, meaning 
                // convolving a signal with a true sinc function would take an infinite amount of time
                // To avoid creating a filter with infinite latency, we have to decide when to cut off
                // our sinc function. We can window (i.e. multiply) our true sinc function with a
                // horizontally stretched sinc function to create a windowed sinc function of our desired width. 
                let lanzcosWindow = Math.sin(x / kernelSize) / (x / kernelSize);

                // A hole exists in the sinc function at zero, special case it
                if (x == 0) {
                    this.kernel[i * kernelSize + j] = 1;
                }
                else {
                    // Apply our window here
                    this.kernel[i * kernelSize + j] = sinc * lanzcosWindow;
                }

                sum += this.kernel[i * kernelSize + j];
            }

            if (normalize) {
                for (let j = 0; j < kernelSize; j++) {
                    this.kernel[i * kernelSize + j] /= sum;
                }
            }
        }
    }

    changeSample(sample) {
        this.sample = sample;
    }

    advance() {
        let convertedSampleRate = this.freqRatio * this.sample.sampleRate;
        this.sampleT += this.secondsPerSample * convertedSampleRate;

        let val0 = this.getSampleDataAt(Math.floor(this.sampleT + 0));
        let val1 = this.getSampleDataAt(Math.floor(this.sampleT + 1));

        let finalVal = val0;
        if (enableAntiAliasing && convertedSampleRate < this.nyquist) {
            let subsampleT = this.sampleT % 1;
            let deltaVal = val1 - val0;
            finalVal = subsampleT < 0.5 ? val0 : val1;
            finalVal += polyBlep((subsampleT + 0.5) % 1, this.secondsPerSample * convertedSampleRate) * deltaVal;
        }

        if (this.sample.enableFilter && filterSamples) {
            this.val = this.filter.transform(finalVal * this.volume);
        } else {
            this.val = finalVal * this.volume;
        }
    }

    getSampleDataAt(sample) {
        if (sample >= this.sample.data.length && this.sample.looping) {
            let sampleNoIntro = sample - this.sample.loopPoint;
            let loopLength = this.sample.data.length - this.sample.loopPoint;
            sampleNoIntro %= loopLength;
            sample = sampleNoIntro + this.sample.loopPoint;
        }

        if (sample < this.sample.data.length) {
            return this.sample.data[sample];
        } else {
            return 0;
        }
    }

    updateFilter() {
        let convertedSampleRate = this.freqRatio * this.sample.sampleRate;
        if (convertedSampleRate > this.nyquist) {
            convertedSampleRate = this.nyquist;
        }
        console.log("cutoff: " + convertedSampleRate);
        this.filter.set(this.sampleRate, convertedSampleRate);
    }

    /** @param {number} midiNote */
    setNote(midiNote) {
        if (elementarySchoolBandMode) {
            midiNote += 0.3 * ((Math.random() - 0.5) * 2);
            // -/+ 30 cents random detune
        }
        this.midiNote = midiNote;
        this.frequency = midiNoteToHz(midiNote);
        this.freqRatio = this.frequency / this.sample.frequency;
        this.updateFilter();
    }

    /** @param {number} semitones */
    setFinetuneLfo(semitones) {
        this.finetuneLfo = semitones;
        this.frequency = midiNoteToHz(this.midiNote + this.finetuneLfo + this.finetune);
        this.freqRatio = this.frequency / this.sample.frequency;
        this.updateFilter();
    }

    setFinetune(semitones) {
        this.finetune = semitones;
        this.frequency = midiNoteToHz(this.midiNote + this.finetuneLfo + this.finetune);
        this.freqRatio = this.frequency / this.sample.frequency;
        this.updateFilter();
    }
}

class SseqController {
    /** @param {Array | Uint8Array} sseqFile
     *  @param {number} dataOffset
     *  @param {MinHeap<Message>} messageBuffer
     **/
    constructor(sseqFile, dataOffset, messageBuffer) {
        this.sseqFile = sseqFile;
        this.dataOffset = dataOffset;
        this.messageBuffer = messageBuffer;

        /** @type {SseqTrack[]} */
        this.tracks = new Array(16);

        for (let i = 0; i < 16; i++) {
            this.tracks[i] = new SseqTrack(this, i);
        }

        this.tracks[0].active = true;
        this.tracks[0].bpm = 120;

        this.destroyed = false;

        this.ticksElapsed = 0;
        this.paused = false;
    }

    tick() {
        if (!this.paused) {
            for (let i = 0; i < 16; i++) {
                if (this.tracks[i].active) {
                    while (this.tracks[i].restingFor == 0) {
                        this.tracks[i].execute();
                    }
                    this.tracks[i].restingFor--;
                }
            }
        }

        this.ticksElapsed++;
    }

    startTrack(num, pc) {
        this.tracks[num].active = true;
        this.tracks[num].pc = pc;
        this.tracks[num].debugLog("Started! PC: " + hexN(pc, 6));
    }

    endTrack(num) {
        this.tracks[num].active = false;
        this.tracks[num].debugLog("Ended track.");
    }
}

class SseqTrack {
    constructor(controller, id) {
        /** @type {SseqController} */
        this.controller = controller;
        this.id = id;

        this.active = false;

        this.bpm = 0;

        this.pc = 0;
        this.pan = 64;
        this.mono = false;
        this.volume = 0;
        this.priority = 0;
        this.program = 0;
        this.bank = 0;

        this.lfoType = 0;
        this.lfoDepth = 0;
        this.lfoRange = 1;
        this.lfoSpeed = 16;
        this.lfoDelay = 0;

        this.pitchBend = 0;
        this.pitchBendRange = 0;

        this.expression = 0;

        this.portamentoEnable = 0;
        this.portamentoTime = 0;

        this.restingFor = 0;

        this.stack = new Uint32Array(64);
        this.sp = 0;
    }

    debugLog(msg) {
        // console.log(`${this.id}: ${msg}`)
    }

    debugLogForce(msg) {
        console.log(`${this.id}: ${msg}`);
    }

    push(val) {
        this.stack[this.sp++] = val;
        if (this.sp >= this.stack.length) alert("SSEQ stack overflow");
    }

    pop() {
        if (this.sp == 0) alert("SSEQ stack underflow");
        return this.stack[--this.sp];
    }

    readPc() {
        return this.controller.sseqFile[this.pc + this.controller.dataOffset];
    }

    readPcInc(bytes = 1) {
        let val = 0;
        for (let i = 0; i < bytes; i++) {
            val |= this.readPc() << (i * 8);
            this.pc++;
        }

        return val;
    }

    readletiableLength(arr, offs) {
        let num = 0;
        for (let i = 0; i < 4; i++) {
            let val = this.readPcInc();

            num <<= 7;
            num |= val & 0x7F;

            if ((val & 0x80) == 0) {
                break;
            }
        }

        return num;
    }

    /** 
     * @param {boolean} fromKeyboard
     * @param {number} type
     * @param {number} param0
     * @param {number} param1
     * @param {number} param2
     */
    sendMessage(fromKeyboard, type, param0 = 0, param1 = 0, param2 = 0) {
        let jitter = 0;
        if (elementarySchoolBandMode) {
            jitter = Math.round((0.005 * Math.random() * 12) * this.controller.tracks[0].bpm);
            console.log(jitter);
        }
        this.controller.messageBuffer.addEntry(new Message(fromKeyboard, this.id, type, param0, param1, param2), this.controller.ticksElapsed + jitter);
    }

    execute() {
        let opcodePc = this.pc;
        let opcode = this.readPcInc();

        if (opcode <= 0x7F) {
            let velocity = this.readPcInc();
            let duration = this.readletiableLength();

            this.debugLog("Note: " + opcode);
            this.debugLog("Velocity: " + velocity);
            this.debugLog("Duration: " + duration);

            this.sendMessage(false, MessageType.PlayNote, opcode, velocity, duration);
        } else {
            switch (opcode) {
                case 0xFE: // Allocate track
                    {
                        // This probably isn't important for emulation
                        let alloced = this.readPcInc(2);

                        for (let i = 0; i < 16; i++) {
                            if (bitTest(alloced, i)) {
                                this.debugLog("Allocated track " + i);
                            }
                        }
                        break;
                    }
                case 0x93: // Start new track thread 
                    {
                        let trackNum = this.readPcInc();
                        let trackOffs = this.readPcInc(3);

                        this.controller.startTrack(trackNum, trackOffs);

                        this.debugLogForce("Started track thread " + trackNum);
                        this.debugLog("Offset: " + hex(trackOffs, 6));

                        break;
                    }
                case 0xC7: // Mono / Poly
                    {
                        let param = this.readPcInc();
                        this.mono = bitTest(param, 0);
                        break;
                    }
                case 0xCE: // Portamento On / Off
                    {
                        this.portamentoEnable = this.readPcInc();
                        this.debugLog("Portamento On / Off: " + this.portamentoEnable);
                        break;
                    }
                case 0xCF: // Portamento Time
                    {
                        this.portamentoTime = this.readPcInc();
                        this.debugLog("Portamento Time: " + this.portamentoTime);
                        break;
                    }
                case 0xE1: // BPM
                    {
                        this.bpm = this.readPcInc(2);
                        this.debugLog("BPM: " + this.bpm);
                        break;
                    }
                case 0xC1: // Volume
                    {
                        this.volume = this.readPcInc();
                        this.sendMessage(false, MessageType.VolumeChange, this.volume);
                        this.debugLog("Volume: " + this.volume);
                        break;
                    }
                case 0x81: // Set bank and program
                    {
                        let bankAndProgram = this.readletiableLength();
                        this.program = bankAndProgram & 0x7F;
                        this.bank = (bankAndProgram >> 7) & 0x7F;

                        this.debugLogForce(`Bank: ${this.bank} Program: ${this.program}`);

                        this.sendMessage(false, MessageType.InstrumentChange, this.bank, this.program);
                        break;
                    }
                case 0xC2: // Master Volume
                    {
                        this.masterVolume = this.readPcInc();
                        this.debugLogForce("Master Volume: " + this.masterVolume);
                        break;
                    }
                case 0xC0: // Pan
                    {
                        this.pan = this.readPcInc();
                        if (this.pan == 127) this.pan = 128;
                        this.debugLog("Pan: " + this.pan);
                        this.sendMessage(false, MessageType.PanChange, this.pan);
                        break;
                    }
                case 0xC6: // Track Priority
                    {
                        this.priority = this.readPcInc();
                        this.debugLog("Track Priority: " + this.priority);
                        break;
                    }
                case 0xC5: // Pitch Bend Range
                    {
                        this.pitchBendRange = this.readPcInc();
                        this.debugLog("Pitch Bend Range: " + this.pitchBendRange);
                        this.sendMessage(false, MessageType.PitchBend);
                        break;
                    }
                case 0xCA: // LFO Depth
                    {
                        this.lfoDepth = this.readPcInc();
                        this.debugLog("LFO Depth: " + this.lfoDepth);
                        break;
                    }
                case 0xCB: // LFO Speed
                    {
                        this.lfoSpeed = this.readPcInc();
                        this.debugLog("LFO Speed: " + this.lfoSpeed);
                        break;
                    }
                case 0xCC: // LFO Type
                    {
                        this.lfoType = this.readPcInc();
                        this.debugLog("LFO Type: " + this.lfoType);
                        if (this.lfoType != LfoType.Volume) {
                            console.warn("Unimplemented LFO type: " + this.lfoType);
                        }
                        break;
                    }
                case 0xCD: // LFO Range
                    {
                        this.lfoRange = this.readPcInc();
                        this.debugLog("LFO Range: " + this.lfoRange);
                        break;
                    }
                case 0xC4: // Pitch Bend
                    {
                        this.pitchBend = this.readPcInc();
                        this.debugLog("Pitch Bend: " + this.pitchBend);
                        this.sendMessage(false, MessageType.PitchBend);
                        break;
                    }
                case 0x80: // Rest
                    {
                        this.restingFor = this.readletiableLength();
                        this.debugLog("Resting For: " + this.restingFor);
                        break;
                    }
                case 0x94: // Jump
                    {
                        let dest = this.readPcInc(3);
                        this.pc = dest;
                        this.debugLogForce(`Jump to: ${hexN(dest, 6)} Tick: ${this.controller.ticksElapsed}`);

                        this.sendMessage(false, MessageType.Jump);
                        break;
                    }
                case 0x95: // Call
                    {
                        let dest = this.readPcInc(3);

                        // Push the return address
                        this.push(this.pc);
                        this.pc = dest;
                        break;
                    }
                case 0xFD: // Return
                    {
                        this.pc = this.pop();
                        break;
                    }
                case 0xB0: // TODO: According to sseq2mid: arithmetic operations?
                    {
                        this.readPcInc(3);
                        break;
                    }
                case 0xE0: // LFO Delay
                    {
                        this.lfoDelay = this.readPcInc(2);
                        this.debugLog("LFO Delay: " + this.lfoDelay);
                        break;
                    }
                case 0xD5: // Expression
                    {
                        this.expression = this.readPcInc();
                        this.debugLog("Expression: " + this.expression);
                        break;
                    }
                case 0xFF: // End of Track
                    {
                        this.controller.endTrack(this.id);
                        this.sendMessage(false, MessageType.TrackEnded);
                        // Set restingFor to non-zero since the controller checks it to stop executing
                        this.restingFor = 1;
                        break;
                    }
                case 0xD0: // Attack Rate
                    {
                        this.attackRate = this.readPcInc();
                        break;
                    }
                case 0xD1: // Decay Rate
                    {
                        this.decayRate = this.readPcInc();
                        break;
                    }
                case 0xD2: // Sustain Rate
                    {
                        this.sustainRate = this.readPcInc();
                        break;
                    }
                case 0xD3: // Release Rate
                    {
                        this.releaseRate = this.readPcInc();
                        break;
                    }
                default:
                    console.error(`${this.id}: Unknown opcode: ` + hex(opcode, 2) + " PC: " + hex(opcodePc, 6));
            }
        }
    }
}

class DelayLine {
    /** @param {number} maxLength */
    constructor(maxLength) {
        this.buffer = new Float64Array(maxLength);
        this.posOut = 0;
        this.delay = 0;
        this.gain = 1;
    }

    /** @param {number} val */
    process(val) {
        this.buffer[(this.posOut + this.delay) % this.buffer.length] = val;
        let outVal = this.buffer[this.posOut];
        this.posOut++;
        if (this.posOut >= this.buffer.length) {
            this.posOut = 0;
        }
        return outVal * this.gain;
    }

    /** @param {number} length */
    setDelay(length) {
        if (length > this.buffer.length) {
            throw "delay length > buffer length";
        }
        this.delay = length;
    }
}

class Synthesizer {
    constructor(sampleRate, instrsAvailable) {
        this.instrsAvailable = instrsAvailable;

        /** @type {SampleInstrument[]} */
        this.instrs = new Array(this.instrsAvailable);
        /** @type {SampleInstrument[]} */
        this.activeInstrs = new Array();
        this.sampleNum = 0;
        this.sampleRate = sampleRate;

        this.valL = 0;
        this.valR = 0;

        this.volume = 1;
        /** @private */
        this.pan = 0.5;

        this.delayLineL = new DelayLine(Math.round(this.sampleRate * 0.1));
        this.delayLineR = new DelayLine(Math.round(this.sampleRate * 0.1));

        this.playingIndex = 0;

        let emptySample = new Sample(new Float32Array(1), 440, sampleRate, false, 0);

        for (let i = 0; i < this.instrs.length; i++) {
            this.instrs[i] = new SampleInstrument(this.sampleRate, emptySample);
        }

        this.finetune = 0;
    }

    /**
     * @param {Sample} sample
     * @param {number} midiNote
     * @param {number} volume
     * @param {number} meta
     */
    play(sample, midiNote, volume, meta) {
        let instr = this.instrs[this.playingIndex];
        if (instr.playing) {
            this.cutInstrument(this.playingIndex);
        }
        instr.sample = sample;
        instr.setNote(midiNote);
        instr.setFinetuneLfo(0);
        instr.setFinetune(this.finetune);
        instr.volume = volume;
        instr.startTime = meta;
        instr.sampleT = 0;
        instr.playing = true;

        let currentIndex = this.playingIndex;

        this.playingIndex++;
        this.playingIndex %= this.instrsAvailable;

        this.activeInstrs.push(instr);

        return currentIndex;
    }

    cutInstrument(instrIndex) {
        const activeInstrIndex = this.activeInstrs.indexOf(this.instrs[instrIndex]);
        if (activeInstrIndex == -1) {
            console.warn("Tried to cut instrument that wasn't playing");
            return;
        }
        this.activeInstrs[activeInstrIndex].playing = false;
        this.activeInstrs.splice(activeInstrIndex, 1);
    }

    nextSample() {
        let valL = 0;
        let valR = 0;

        for (const instr of this.activeInstrs) {
            instr.advance();
            let val = instr.val;
            valL += val * (1 - this.pan);
            valR += val * this.pan;
        }

        if (enableStereoSeparation) {
            this.valL = this.delayLineL.process(valL) * this.volume;
            this.valR = this.delayLineR.process(valR) * this.volume;
        } else {
            this.valL = valL * this.volume;
            this.valR = valR * this.volume;
        }

        this.sampleNum++;
    }

    setFinetune(semitones) {
        this.finetune = semitones;
        for (let instr of this.instrs) {
            instr.setFinetune(semitones);
        }
    }

    /** @param {number} pan */
    setPan(pan) {
        const SPEED_OF_SOUND = 343; // meters per second
        // let's pretend panning moves the sound source in a semicircle around and in front of the listener
        let r = 3; // semicircle radius
        let earX = 0.20; // absolute position of ears on the X axis
        let x = pan * 2 - 1; // [0, 1] -> [-1, -1]
        // force stereo separation on barely panned channels
        let gainR = 1;
        if (enableForceStereoSeparation) {
            if (x > -0.2 && x < 0.2) {
                // gainR = -1;
                x = 0.2 * Math.sign(x);
            }
        }
        let y = Math.sqrt((r ** 2) - x ** 2);
        let distL = Math.sqrt((earX + x) ** 2 + y ** 2);
        let distR = Math.sqrt((-earX + x) ** 2 + y ** 2);
        let minDist = Math.min(distL, distR);
        distL -= minDist;
        distR -= minDist;
        let delaySL = distL / SPEED_OF_SOUND * 50;
        let delaySR = distR / SPEED_OF_SOUND * 50;
        let delayL = Math.round(delaySL * this.sampleRate);
        let delayR = Math.round(delaySR * this.sampleRate);
        console.log(`L:${delaySL * 1000}ms R:${delaySR * 1000}ms X:${x}`);
        this.delayLineL.setDelay(delayL);
        this.delayLineR.setDelay(delayR);
        this.delayLineR.gain = gainR;
        this.pan = pan;
    }
}

/** @template T */
class MinHeapEntry {
    data /** @type {T} */;
    index = 0;
    priority = 0;
}

/** @template T */
class MinHeap {
    static parent(n) { return (n - 1) >> 1; }
    static leftChild(n) { return n * 2 + 1; }
    static rightChild(n) { return n * 2 + 2; }

    /** @param {number} size */
    constructor(size) {
        /** @type {MinHeapEntry<T>[]} */
        this.heap = new Array(size);

        for (let i = 0; i < this.heap.length; i++) {
            this.heap[i] = new MinHeapEntry();
            this.heap[i].index = i;
        }
    }

    entries = 0;

    static createEmptyEntry() {
        return new MinHeapEntry();
    }

    addEntry(data, priority) {
        if (this.entries >= this.heap.length) {
            alert("Heap overflow!");
            return;
        }

        let index = this.entries;
        this.entries++;
        this.heap[index].data = data;
        this.heap[index].priority = priority;
        this.heap[index].index = index;

        while (index != 0) {
            let parentIndex = MinHeap.parent(index);
            if (this.heap[parentIndex].priority > this.heap[index].priority) {
                this.swap(index, parentIndex);
                index = parentIndex;
            } else {
                break;
            }
        }
        this.updateNextEvent();
    }

    updateNextEvent() {
        if (this.entries > 0) {
            this.nextEventTicks = this.heap[0].priority;
        }
    }

    getFirstEntry() {
        if (this.entries <= 0) {
            alert("Tried to get from empty heap!");
            return this.heap[0]; // This isn't supposed to happen.
        }

        return this.heap[0];
    }

    returnEvent = MinHeap.createEmptyEntry();

    popFirstEntry() {
        let event = this.getFirstEntry();

        this.returnEvent.data = event.data;
        this.returnEvent.priority = event.priority;
        this.returnEvent.index = event.index;

        if (this.entries == 1) {
            this.entries--;
            return this.returnEvent;
        }

        this.swap(0, this.entries - 1);

        this.entries--;

        // Satisfy the heap property again
        let index = 0;
        while (true) {
            let left = MinHeap.leftChild(index);
            let right = MinHeap.rightChild(index);
            let smallest = index;

            if (left < this.entries && this.heap[left].priority < this.heap[index].priority) {
                smallest = left;
            }
            if (right < this.entries && this.heap[right].priority < this.heap[smallest].priority) {
                smallest = right;
            }

            if (smallest != index) {
                this.swap(index, smallest);
                index = smallest;
            } else {
                break;
            }
        }

        this.updateNextEvent();
        return this.returnEvent;
    }

    setPriorityLower(index, newVal) {
        this.heap[index].priority = newVal;

        while (index != 0) {
            let parentIndex = MinHeap.parent(index);
            if (this.heap[parentIndex].priority > this.heap[index].priority) {
                this.swap(index, parentIndex);
                index = parentIndex;
            } else {
                break;
            }
        }
    }

    deleteEvent(index) {
        this.setPriorityLower(index, -9999);
        this.popFirstEntry();
    }

    swap(ix, iy) {
        // console.log(`Swapped ${ix} with ${iy}`);
        let temp = this.heap[ix];
        this.heap[ix] = this.heap[iy];
        this.heap[ix].index = ix;
        this.heap[iy] = temp;
        this.heap[iy].index = iy;
    }
}

const AdsrState = {
    Attack: 0,
    Decay: 1,
    Sustain: 2,
    Release: 3,
};

// from pret/pokediamond
const sAttackCoeffTable = [
    0, 1, 5, 14, 26, 38, 51, 63, 73, 84, 92, 100, 109, 116, 123, 127, 132, 137, 143, 0,
];

const SNDi_DecibelSquareTable = [
    -32768, -722, -721, -651, -601, -562, -530, -503,
    -480, -460, -442, -425, -410, -396, -383, -371,
    -360, -349, -339, -330, -321, -313, -305, -297,
    -289, -282, -276, -269, -263, -257, -251, -245,
    -239, -234, -229, -224, -219, -214, -210, -205,
    -201, -196, -192, -188, -184, -180, -176, -173,
    -169, -165, -162, -158, -155, -152, -149, -145,
    -142, -139, -136, -133, -130, -127, -125, -122,
    -119, -116, -114, -111, -109, -106, -103, -101,
    -99, -96, -94, -91, -89, -87, -85, -82,
    -80, -78, -76, -74, -72, -70, -68, -66,
    -64, -62, -60, -58, -56, -54, -52, -50,
    -49, -47, -45, -43, -42, -40, -38, -36,
    -35, -33, -31, -30, -28, -27, -25, -23,
    -22, -20, -19, -17, -16, -14, -13, -11,
    -10, -8, -7, -6, -4, -3, -1, 0,
];

// this table is located in the DS ARM7 BIOS, copied from desmume
const getvoltbl = [
    0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
    0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
    0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03,
    0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03,
    0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04,
    0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04,
    0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05,
    0x05, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x08, 0x08, 0x08, 0x08,
    0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09,
    0x09, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0B, 0x0B, 0x0B, 0x0B, 0x0B, 0x0B, 0x0B,
    0x0B, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0E,
    0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x10, 0x10, 0x10, 0x10, 0x10,
    0x10, 0x11, 0x11, 0x11, 0x11, 0x11, 0x12, 0x12, 0x12, 0x12, 0x12, 0x13, 0x13, 0x13, 0x13, 0x14,
    0x14, 0x14, 0x14, 0x14, 0x15, 0x15, 0x15, 0x15, 0x16, 0x16, 0x16, 0x16, 0x17, 0x17, 0x17, 0x18,
    0x18, 0x18, 0x18, 0x19, 0x19, 0x19, 0x19, 0x1A, 0x1A, 0x1A, 0x1B, 0x1B, 0x1B, 0x1C, 0x1C, 0x1C,
    0x1D, 0x1D, 0x1D, 0x1E, 0x1E, 0x1E, 0x1F, 0x1F, 0x1F, 0x20, 0x20, 0x20, 0x21, 0x21, 0x22, 0x22,
    0x22, 0x23, 0x23, 0x24, 0x24, 0x24, 0x25, 0x25, 0x26, 0x26, 0x27, 0x27, 0x27, 0x28, 0x28, 0x29,
    0x29, 0x2A, 0x2A, 0x2B, 0x2B, 0x2C, 0x2C, 0x2D, 0x2D, 0x2E, 0x2E, 0x2F, 0x2F, 0x30, 0x31, 0x31,
    0x32, 0x32, 0x33, 0x33, 0x34, 0x35, 0x35, 0x36, 0x36, 0x37, 0x38, 0x38, 0x39, 0x3A, 0x3A, 0x3B,
    0x3C, 0x3C, 0x3D, 0x3E, 0x3F, 0x3F, 0x40, 0x41, 0x42, 0x42, 0x43, 0x44, 0x45, 0x45, 0x46, 0x47,
    0x48, 0x49, 0x4A, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x4F, 0x50, 0x51, 0x52, 0x52, 0x53, 0x54, 0x55,
    0x56, 0x57, 0x58, 0x59, 0x5A, 0x5B, 0x5D, 0x5E, 0x5F, 0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x67,
    0x68, 0x69, 0x6A, 0x6B, 0x6D, 0x6E, 0x6F, 0x71, 0x72, 0x73, 0x75, 0x76, 0x77, 0x79, 0x7A, 0x7B,
    0x7D, 0x7E, 0x7F, 0x20, 0x21, 0x21, 0x21, 0x22, 0x22, 0x23, 0x23, 0x23, 0x24, 0x24, 0x25, 0x25,
    0x26, 0x26, 0x26, 0x27, 0x27, 0x28, 0x28, 0x29, 0x29, 0x2A, 0x2A, 0x2B, 0x2B, 0x2C, 0x2C, 0x2D,
    0x2D, 0x2E, 0x2E, 0x2F, 0x2F, 0x30, 0x30, 0x31, 0x31, 0x32, 0x33, 0x33, 0x34, 0x34, 0x35, 0x36,
    0x36, 0x37, 0x37, 0x38, 0x39, 0x39, 0x3A, 0x3B, 0x3B, 0x3C, 0x3D, 0x3E, 0x3E, 0x3F, 0x40, 0x40,
    0x41, 0x42, 0x43, 0x43, 0x44, 0x45, 0x46, 0x47, 0x47, 0x48, 0x49, 0x4A, 0x4B, 0x4C, 0x4D, 0x4D,
    0x4E, 0x4F, 0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x5B, 0x5C, 0x5D,
    0x5E, 0x5F, 0x60, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x69, 0x6A, 0x6B, 0x6C, 0x6D, 0x6F, 0x70,
    0x71, 0x73, 0x74, 0x75, 0x77, 0x78, 0x79, 0x7B, 0x7C, 0x7E, 0x7E, 0x40, 0x41, 0x42, 0x43, 0x43,
    0x44, 0x45, 0x46, 0x47, 0x47, 0x48, 0x49, 0x4A, 0x4B, 0x4C, 0x4C, 0x4D, 0x4E, 0x4F, 0x50, 0x51,
    0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x5B, 0x5C, 0x5D, 0x5E, 0x5F, 0x60, 0x61,
    0x62, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6B, 0x6C, 0x6D, 0x6E, 0x70, 0x71, 0x72, 0x74, 0x75,
    0x76, 0x78, 0x79, 0x7B, 0x7C, 0x7D, 0x7E, 0x40, 0x41, 0x42, 0x42, 0x43, 0x44, 0x45, 0x46, 0x46,
    0x47, 0x48, 0x49, 0x4A, 0x4B, 0x4B, 0x4C, 0x4D, 0x4E, 0x4F, 0x50, 0x51, 0x52, 0x53, 0x54, 0x55,
    0x56, 0x57, 0x58, 0x59, 0x5A, 0x5B, 0x5C, 0x5D, 0x5E, 0x5F, 0x60, 0x61, 0x62, 0x63, 0x65, 0x66,
    0x67, 0x68, 0x69, 0x6A, 0x6C, 0x6D, 0x6E, 0x6F, 0x71, 0x72, 0x73, 0x75, 0x76, 0x77, 0x79, 0x7A,
    0x7C, 0x7D, 0x7E, 0x7F
];

const squares = [
    new Sample(new Float32Array([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5]), 1, 8, true, 0),
    new Sample(new Float32Array([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, 0.5]), 1, 8, true, 0),
    new Sample(new Float32Array([-0.5, -0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5]), 1, 8, true, 0),
    new Sample(new Float32Array([-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5]), 1, 8, true, 0),
    new Sample(new Float32Array([-0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), 1, 8, true, 0),
    new Sample(new Float32Array([-0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), 1, 8, true, 0),
    new Sample(new Float32Array([-0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), 1, 8, true, 0),
    new Sample(new Float32Array([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]), 1, 8, true, 0)
];

// based off SND_CalcChannelVolume from pret/pokediamond
function calcChannelVolume(velocity, adsrTimer) {
    const SND_VOL_DB_MIN = -723;

    let vol = 0;

    vol += SNDi_DecibelSquareTable[velocity];
    vol += adsrTimer >> 7;

    if (vol < SND_VOL_DB_MIN) {
        vol = SND_VOL_DB_MIN;
    } else if (vol > 0) {
        vol = 0;
    }

    let result = getvoltbl[vol - SND_VOL_DB_MIN];

    if (vol < -240)
        result /= 16;
    else if (vol < -120)
        result /= 4;
    else if (vol < -60)
        result /= 2;
    else
        result /= 1;

    return result / 127;
}

class FsVisControllerBridge {
    /** @param {Sdat} sdat */
    /** @param {string} id */
    /** @param {number} runAheadTicks */
    constructor(sdat, id, runAheadTicks) {
        this.runAheadTicks = runAheadTicks;

        let info = sdat.sseqInfos[id];
        let file = sdat.fat[info.fileId];
        let dataOffset = read32LE(file, 0x18);

        /** @type {MinHeap<Message>} */
        this.messageBuffer = new MinHeap(512);
        this.controller = new SseqController(file, dataOffset, this.messageBuffer);
        /** @type {CircularBuffer<Message>} */
        this.activeNotes = new CircularBuffer(2048);

        this.bpmTimer = 0;

        for (let i = 0; i < runAheadTicks; i++) {
            this.tick();
        }
    }

    tick() {
        this.bpmTimer += this.controller.tracks[0].bpm;
        while (this.bpmTimer >= 240) {
            this.bpmTimer -= 240;

            this.controller.tick();

            while (this.messageBuffer.entries > 0 &&
                this.messageBuffer.getFirstEntry().priority <= this.controller.ticksElapsed) {
                /** @type {Message} */
                let msg = this.messageBuffer.popFirstEntry().data;

                switch (msg.type) {
                    case MessageType.PlayNote:
                        if (this.activeNotes.entries >= this.activeNotes.size) {
                            this.activeNotes.pop();
                        }

                        msg.timestamp = this.controller.ticksElapsed;
                        this.activeNotes.insert(msg);
                        break;
                }
            }
        }
    }
}

const LfoType = {
    Pitch: 0,
    Volume: 1,
    Pan: 2
};

// pret/pokediamond
const sLfoSinTable = [
    0,
    6,
    12,
    19,
    25,
    31,
    37,
    43,
    49,
    54,
    60,
    65,
    71,
    76,
    81,
    85,
    90,
    94,
    98,
    102,
    106,
    109,
    112,
    115,
    117,
    120,
    122,
    123,
    125,
    126,
    126,
    127,
    127,
    0,
    0,
    0
];

// pret/pokediamond
function SND_SinIdx(x) {
    if (x < 0x20) {
        return sLfoSinTable[x];
    }
    else if (x < 0x40) {
        return sLfoSinTable[0x40 - x];
    }
    else if (x < 0x60) {
        return (-sLfoSinTable[x - 0x40]) << 24 >> 24;
    }
    else {
        return (-sLfoSinTable[0x20 - (x - 0x60)]) << 24 >> 24;
    }
}

class ControllerBridge {
    /** @param {number} sampleRate */
    /** @param {Sdat} sdat */
    /** @param {string} id */
    constructor(sampleRate, sdat, id) {
        let info = sdat.sseqInfos[id];
        this.bankInfo = sdat.sbnkInfos[info.bank];
        this.bank = sdat.banks[info.bank];

        console.log("Playing SSEQ Id:" + id);
        console.log("FAT ID:" + info.fileId);

        console.log(`Linked archives: ${this.bankInfo.swarId[0]} ${this.bankInfo.swarId[1]} ${this.bankInfo.swarId[2]} ${this.bankInfo.swarId[3]}`);
        let file = sdat.fat[info.fileId];

        for (let i = 0; i < this.bank.instruments.length; i++) {
            let instrument = this.bank.instruments[i];
            let typeString = "";
            switch (instrument.fRecord) {
                case InstrumentType.Drumset:
                    typeString = "Drumset";
                    break;
                case InstrumentType.MultiSample:
                    typeString = "Multi-Sample Instrument";
                    break;
                case InstrumentType.PsgNoise:
                    typeString = "PSG Noise";
                    break;
                case InstrumentType.PsgPulse:
                    typeString = "PSG Pulse";
                    break;
                case InstrumentType.SingleSample:
                    typeString = "Single-Sample Instrument";
                    break;

            }

            if (instrument.fRecord != 0) {
                console.log(`Program ${i}: ${typeString}\nLinked archive ${instrument.swarInfoId[0]} Sample ${instrument.swavInfoId[0]}`);
            }
        }

        let dataOffset = read32LE(file, 0x18);
        if (dataOffset != 0x1C) alert("SSEQ offset is not 0x1C? it is: " + hex(dataOffset, 8));

        this.sdat = sdat;
        /** @type {MinHeap<Message>} */
        this.messageBuffer = new MinHeap(1024);
        this.controller = new SseqController(file, dataOffset, this.messageBuffer);

        /** @type {Uint8Array[]} */
        this.notesOn = [];
        for (let i = 0; i < 16; i++) {
            this.notesOn[i] = new Uint8Array(128);
        }

        /** @type {Synthesizer[]} */
        this.synthesizers = new Array(16);
        for (let i = 0; i < 16; i++) {
            this.synthesizers[i] = new Synthesizer(sampleRate, 16);
        }
        this.destroyed = false;

        this.jumps = 0;
        this.fadingStart = false;

        this.loop = 0;

        // /** @type {MinHeap<}
        this.activeNoteData = new MinHeap(1024);

        this.bpmTimer = 0;

        this.activeKeyboardTrackNum = null;
    }

    tick() {
        for (let i = 0; i < this.activeNoteData.entries; i++) {
            let entry = this.activeNoteData.heap[i];
            let data = entry.data;
            /** @type {InstrumentRecord} */
            let instrument = data.instrument;

            let instr = this.synthesizers[entry.data.trackNum].instrs[entry.data.synthInstrIndex];
            // sometimes a SampleInstrument will be reused before the note it is playing is over due to Synthesizer polyphony limits
            // check here to make sure the note entry stored in the heap is referring to the same note it originally did 
            if (instr.startTime == entry.data.startTime && instr.playing) {
                if (this.controller.ticksElapsed >= entry.priority && !entry.data.fromKeyboard) {
                    if (data.adsrState != AdsrState.Release) {
                        // console.log("to release: " + instrument.release[data.instrumentEntryIndex]);
                        this.notesOn[entry.data.trackNum][entry.data.midiNote] = 0;
                        data.adsrState = AdsrState.Release;
                    }
                }

                // LFO code based off pret/pokediamond
                let track = this.controller.tracks[data.trackNum];
                let lfoValue;
                if (track.lfoDepth == 0) {
                    lfoValue = BigInt(0);
                } else if (data.lfoDelayCounter < track.lfoDelay) {
                    lfoValue = BigInt(0);
                } else {
                    lfoValue = BigInt(SND_SinIdx(data.lfoCounter >>> 8) * track.lfoDepth * track.lfoRange);
                }

                if (lfoValue != 0n) {
                    switch (track.lfoType) {
                        case LfoType.Volume:
                            lfoValue *= 60n;
                            break;
                        case LfoType.Pitch:
                            lfoValue <<= 6n;
                            break;
                        case LfoType.Pan:
                            lfoValue <<= 6n;
                            break;
                    }
                    lfoValue >>= 14n;
                }

                if (data.delayCounter < track.lfoDelay) {
                    data.delayCounter++;
                } else {
                    let tmp = data.lfoCounter;
                    tmp += track.lfoSpeed << 6;
                    tmp >>>= 8;
                    while (tmp >= 0x80) {
                        tmp -= 0x80;
                    }
                    data.lfoCounter += track.lfoSpeed << 6;
                    data.lfoCounter &= 0xFF;
                    data.lfoCounter |= tmp << 8;

                    if (lfoValue != 0n) {
                        switch (track.lfoType) {
                            case LfoType.Pitch:
                                // LFO value is in 1/64ths of a semitone
                                instr.setFinetuneLfo(Number(lfoValue) / 64);
                                break;
                            default:
                                break;
                        }
                    }
                }

                // all thanks to ipatix at pret/pokediamond
                switch (data.adsrState) {
                    case AdsrState.Attack:
                        data.adsrTimer = -((-instrument.attackCoefficient[data.instrumentEntryIndex] * data.adsrTimer) >> 8);
                        // console.log(data.adsrTimer);
                        instr.volume = calcChannelVolume(data.velocity, data.adsrTimer);
                        // one instrument hits full volume, start decay
                        if (data.adsrTimer == 0) {
                            // console.log("to decay: " + instrument.decay[data.instrumentEntryIndex])
                            data.adsrState = AdsrState.Decay;
                        }
                        break;
                    case AdsrState.Decay:
                        data.adsrTimer -= instrument.decayCoefficient[data.instrumentEntryIndex];
                        // console.log(data.adsrTimer);
                        // when instrument decays to sustain volume, go into sustain state

                        if (data.adsrTimer <= instrument.sustainLevel[data.instrumentEntryIndex]) {
                            // console.log("to sustain:  " + instrument.sustain[data.instrumentEntryIndex]);
                            // console.log("vol: " + (12 8 + data.adsrTimer / 723));
                            data.adsrTimer = instrument.sustainLevel[data.instrumentEntryIndex];
                            data.adsrState = AdsrState.Sustain;
                        }

                        instr.volume = calcChannelVolume(data.velocity, data.adsrTimer);
                        break;
                    case AdsrState.Sustain:
                        break;
                    case AdsrState.Release:
                        if (data.adsrTimer <= -92544 || instrument.fRecord == InstrumentType.PsgPulse) {
                            this.synthesizers[data.trackNum].cutInstrument(entry.data.synthInstrIndex);
                            data.scheduledForDeletion = true;
                        } else {
                            data.adsrTimer -= instrument.releaseCoefficient[data.instrumentEntryIndex];
                            instr.volume = calcChannelVolume(data.velocity, data.adsrTimer);
                        }
                        break;
                }
            }
        }

        // remove stale entries from heap
        if (this.activeNoteData.entries > 0) {
            let entry = this.activeNoteData.getFirstEntry();
            let instr = this.synthesizers[entry.data.trackNum].instrs[entry.data.synthInstrIndex];

            // check instr.startTime != entry.data.startTime to remove entries for 
            // SampleInstruments that were reused early because of Synthesizer polyphony limits
            if (entry.data.scheduledForDeletion || instr.startTime != entry.data.startTime) {
                this.activeNoteData.popFirstEntry();
            }
        }

        this.bpmTimer += this.controller.tracks[0].bpm;
        while (this.bpmTimer >= 240) {
            this.bpmTimer -= 240;

            this.controller.tick();

            while (this.messageBuffer.entries > 0 &&
                this.messageBuffer.getFirstEntry().priority <= this.controller.ticksElapsed) {

                /** @type {Message} */
                let msg = this.messageBuffer.popFirstEntry().data;

                switch (msg.type) {
                    case MessageType.PlayNote:
                        if (this.activeKeyboardTrackNum != msg.trackNum || msg.fromKeyboard) {
                            let midiNote = msg.param0;
                            let velocity = msg.param1;
                            let duration = msg.param2;

                            if (midiNote < 21 || midiNote > 108) console.log("MIDI note out of piano range: " + midiNote);

                            // The archive ID inside each instrument record inside each SBNK file
                            // refers to the archive ID referred to by the corresponding SBNK entry in the INFO block

                            /** @type {InstrumentRecord} */
                            let instrument = this.bank.instruments[this.controller.tracks[msg.trackNum].program];

                            let index = instrument.resolveEntryIndex(midiNote);
                            let archiveId = instrument.swarInfoId[index];
                            let sampleId = instrument.swavInfoId[index];

                            let archive = this.sdat.sampleArchives[this.bankInfo.swarId[archiveId]];
                            if (archive) {
                                let sample = archive[sampleId];

                                if (instrument.fRecord == InstrumentType.PsgPulse) {
                                    sample = squares[sampleId];
                                    sample.enableFilter = false;
                                } else {
                                    sample.frequency = midiNoteToHz(instrument.noteNumber[index]);
                                    sample.enableFilter = true;
                                }

                                // if (msg.fromKeyboard) { 
                                // console.log(this.bank);
                                // console.log("Program " + this.controller.tracks[msg.trackNum].program);
                                // console.log("MIDI Note " + midiNote);
                                // console.log("Base MIDI Note: " + instrument.noteNumber[index]);

                                // if (instrument.fRecord == InstrumentType.PsgPulse) {
                                //     console.log("PSG Pulse");
                                // }

                                // console.log("Attack: " + instrument.attack[index]);
                                // console.log("Decay: " + instrument.decay[index]);
                                // console.log("Sustain: " + instrument.sustain[index]);
                                // console.log("Release: " + instrument.release[index]);

                                // console.log("Attack Coefficient: " + instrument.attackCoefficient[index]);
                                // console.log("Decay Coefficient: " + instrument.decayCoefficient[index]);
                                // console.log("Sustain Level: " + instrument.sustainLevel[index]);
                                // console.log("Release Coefficient: " + instrument.releaseCoefficient[index]);

                                // }

                                // TODO: implement per-instrument pan

                                let track = this.controller.tracks[msg.trackNum];
                                let initialVolume = instrument.attackCoefficient[index] == 0 ? calcChannelVolume(velocity, 0) : 0;
                                let synthInstrIndex = this.synthesizers[msg.trackNum].play(sample, midiNote, initialVolume, this.controller.ticksElapsed);

                                this.notesOn[msg.trackNum][midiNote] = 1;
                                this.activeNoteData.addEntry(
                                    {
                                        trackNum: msg.trackNum,
                                        midiNote: midiNote,
                                        velocity: velocity,
                                        synthInstrIndex: synthInstrIndex,
                                        startTime: this.controller.ticksElapsed,
                                        instrument: instrument,
                                        instrumentEntryIndex: index,
                                        adsrState: AdsrState.Attack,
                                        adsrTimer: -92544, // idk why this number, ask gbatek
                                        fromKeyboard: msg.fromKeyboard,
                                        lfoCounter: 0,
                                        lfoDelayCounter: 0
                                    },
                                    this.controller.ticksElapsed + duration
                                );
                            }
                        }
                        break;
                    case MessageType.Jump: {
                        this.jumps++;
                        break;
                    }
                    case MessageType.TrackEnded: {
                        let tracksActive = 0;
                        for (let i = 0; i < 16; i++) {
                            if (this.controller.tracks[i].active) {
                                tracksActive++;
                            }
                        }

                        if (tracksActive == 0) {
                            this.fadingStart = true;
                        }
                        break;
                    }
                    case MessageType.VolumeChange: {
                        this.synthesizers[msg.trackNum].volume = msg.param0 / 127;
                        break;
                    }
                    case MessageType.PanChange: {
                        this.synthesizers[msg.trackNum].setPan(msg.param0 / 128);
                        break;
                    }
                    case MessageType.PitchBend: {
                        let track = this.controller.tracks[msg.trackNum];
                        let pitchBend = track.pitchBend << 25 >> 25; // sign extend
                        pitchBend *= track.pitchBendRange / 2;
                        // pitch bend specified in 1/64 of a semitone
                        this.synthesizers[msg.trackNum].setFinetune(pitchBend / 64);
                        break;
                    }
                }
            }
        }

    }

    destroy() {
        this.destroyed = true;
    }
}

function bitTest(i, bit) {
    return (i & (1 << bit)) !== 0;
}

/**
* @param {Sdat} sdat
* @param {number} id
*/
function playSeqById(sdat, id) {
    playSeq(sdat, sdat.sseqIdNameDict[id]);
}

/**
 * @param {Sdat} sdat
 * @param {string} name
 */
async function playSeq(sdat, name) {
    currentlyPlayingSdat = sdat;
    currentlyPlayingName = name;
    if (currentBridge) {
        currentBridge.destroy();
        currentPlayer?.ctx.close();
    }

    const BUFFER_SIZE = 1024;
    const SAMPLE_RATE = 32768;

    let id = sdat.sseqNameIdDict[name];

    currentlyPlayingId = id;

    let bufferL = new Float32Array(BUFFER_SIZE);
    let bufferR = new Float32Array(BUFFER_SIZE);

    let fsVisBridge = new FsVisControllerBridge(sdat, id, 384 * 5);
    let bridge = new ControllerBridge(SAMPLE_RATE, sdat, id);

    // // debugging hexdump
    // let offs = 0;
    // for (let i = 0; i < 16; i++) {
    //     let str = "";
    //     for (let j = 0; j < 16; j++) {
    //         str += hexN(file[offs], 2) + " ";
    //         offs++;
    //     }
    //     console.log(str);
    // }
    currentBridge = bridge;
    currentFsVisBridge = fsVisBridge;

    let timer = 0;
    function synthesizeMore() {
        for (let i = 0; i < BUFFER_SIZE; i++) {
            // nintendo DS clock speed
            timer += 33513982;
            while (timer >= 64 * 2728 * SAMPLE_RATE) {
                timer -= 64 * 2728 * SAMPLE_RATE;

                bridge.tick();
                fsVisBridge.tick();
            }
            // synthesizer.play(440);

            let valL = 0;
            let valR = 0;
            for (let i = 0; i < 16; i++) {
                bridge.synthesizers[i].nextSample();
                if (trackEnables[i]) {
                    valL += bridge.synthesizers[i].valL;
                    valR += bridge.synthesizers[i].valR;
                }
            }

            bufferL[i] = valL;
            bufferR[i] = valR;

            if (bridge.controller.destroyed) {
                return;
            }
        }

        // synthesizer.play(Math.random() * 880);

        // console.log(inBufferPos);

        player.queueAudio(bufferL, bufferR);

        // console.log("Syntheszing more audio");
    }

    let player = new AudioPlayer(BUFFER_SIZE, SAMPLE_RATE, synthesizeMore);
    currentPlayer = player;
    synthesizeMore();
}

/**
 * @param {Sample} sample 
 */
async function downloadSample(sample) {
    let totalSamples = 0;
    let downloader = new WavEncoder(sample.sampleRate, 16);
    for (let i = 0; i < sample.data.length; i++) {
        let val = sample.data[i];
        downloader.addSample(val, val);
        totalSamples++;
    }

    for (let i = 0; i < 2; i++) {
        let pos = sample.loopPoint;
        console.log(totalSamples);
        while (pos < sample.data.length) {
            let val = sample.data[pos++];
            downloader.addSample(val, val);
            totalSamples++;
        }
    }

    downloadUint8Array("sample.wav", downloader.encode());
}

/**
* @param {Sdat} sdat
* @param {string} name
*/
async function renderAndDownloadSeq(sdat, name) {
    const OVERSAMPLE = 4;
    const SAMPLE_RATE = 32768;

    let id = sdat.sseqNameIdDict[name];

    let bridge = new ControllerBridge(SAMPLE_RATE * OVERSAMPLE, sdat, id);

    console.log("Rendering SSEQ Id:" + id);
    // console.log("FAT ID:" + info.fileId);

    let encoder = new WavEncoder(SAMPLE_RATE, 16);

    let sample = 0;
    let fadingOut = false;
    let fadeoutStartSample = 0;
    let loop = 0;

    let timer = 0;
    let playing = true;
    let fadeoutLength = 10; // in seconds 

    // keep it under 480 seconds
    while (playing && sample < SAMPLE_RATE * 480) {
        // nintendo DS clock speed
        timer += 33513982;
        while (timer >= 64 * 2728 * SAMPLE_RATE) {
            timer -= 64 * 2728 * SAMPLE_RATE;

            bridge.tick();
        }

        if (bridge.jumps > 0) {
            bridge.jumps = 0;
            loop++;

            if (loop == 2) {
                bridge.fadingStart = true;
            }
        }

        if (bridge.fadingStart) {
            bridge.fadingStart = false;
            fadingOut = true;
            fadeoutStartSample = sample + SAMPLE_RATE * 2;
            console.log("Starting fadeout at sample: " + fadeoutStartSample);
        }

        let fadeoutVolMul = 1;

        if (fadingOut) {
            let fadeoutSample = sample - fadeoutStartSample;
            if (fadeoutSample >= 0) {
                let fadeoutTime = fadeoutSample / SAMPLE_RATE;

                let ratio = fadeoutTime / fadeoutLength;

                fadeoutVolMul = 1 - ratio;

                if (fadeoutVolMul <= 0) {
                    playing = false;
                }
            }
        }

        let valL = 0;
        let valR = 0;
        for (let i = 0; i < 16; i++) {
            if (trackEnables[i]) {
                let synth = bridge.synthesizers[i];
                for (let j = 0; j < 4; j++) {
                    synth.nextSample();
                    valL += synth.valL;
                    valR += synth.valR;
                }
            }
        }
        valL /= OVERSAMPLE;
        valR /= OVERSAMPLE;

        encoder.addSample(valL * 0.5 * fadeoutVolMul, valR * 0.5 * fadeoutVolMul);

        sample++;
    }

    downloadUint8Array(name + ".wav", encoder.encode());
}

function clamp(val, min, max) {
    return Math.min(Math.max(val, min), max);
}

function decodePcm8(pcm8Data) {
    let out = new Float32Array(pcm8Data.length);

    for (let i = 0; i < out.length; i++) {
        out[i] = (pcm8Data[i] << 24 >> 24) / 128;
    }

    return out;
}

function decodePcm16(pcm16Data) {
    let out = new Float32Array(pcm16Data.length >> 1);

    for (let i = 0; i < out.length; i++) {
        out[i] = ((read16LE(pcm16Data, i * 2) << 16) >> 16) / 32768;
    }

    return out;
}

const indexTable = [-1, -1, -1, -1, 2, 4, 6, 8];
const adpcmTable = [
    0x0007, 0x0008, 0x0009, 0x000A, 0x000B, 0x000C, 0x000D, 0x000E, 0x0010, 0x0011, 0x0013, 0x0015,
    0x0017, 0x0019, 0x001C, 0x001F, 0x0022, 0x0025, 0x0029, 0x002D, 0x0032, 0x0037, 0x003C, 0x0042,
    0x0049, 0x0050, 0x0058, 0x0061, 0x006B, 0x0076, 0x0082, 0x008F, 0x009D, 0x00AD, 0x00BE, 0x00D1,
    0x00E6, 0x00FD, 0x0117, 0x0133, 0x0151, 0x0173, 0x0198, 0x01C1, 0x01EE, 0x0220, 0x0256, 0x0292,
    0x02D4, 0x031C, 0x036C, 0x03C3, 0x0424, 0x048E, 0x0502, 0x0583, 0x0610, 0x06AB, 0x0756, 0x0812,
    0x08E0, 0x09C3, 0x0ABD, 0x0BD0, 0x0CFF, 0x0E4C, 0x0FBA, 0x114C, 0x1307, 0x14EE, 0x1706, 0x1954,
    0x1BDC, 0x1EA5, 0x21B6, 0x2515, 0x28CA, 0x2CDF, 0x315B, 0x364B, 0x3BB9, 0x41B2, 0x4844, 0x4F7E,
    0x5771, 0x602F, 0x69CE, 0x7462, 0x7FFF
];

// Decodes IMA-ADPCM to PCM16
function decodeAdpcm(adpcmData) {
    let out = new Float32Array((adpcmData.length - 4) * 2);
    let outOffs = 0;

    let adpcmIndex = 0;

    let header = read32LE(adpcmData, 0);
    // ADPCM header
    let currentValue = header & 0xFFFF;
    adpcmIndex = clamp(header >> 16, 0, 88);

    for (let i = 4; i < adpcmData.length; i++) {
        for (let j = 0; j < 2; j++) {
            let data = (adpcmData[i] >> (j * 4)) & 0xF;

            let tableVal = adpcmTable[adpcmIndex];
            let diff = tableVal >> 3;
            if ((data & 1) != 0) diff += tableVal >> 2;
            if ((data & 2) != 0) diff += tableVal >> 1;
            if ((data & 4) != 0) diff += tableVal >> 0;

            if ((data & 8) == 8) {
                currentValue = Math.max(currentValue - diff, -0x7FFF);
            }
            else {
                currentValue = Math.min(currentValue + diff, 0x7FFF);
            }
            adpcmIndex = clamp(adpcmIndex + indexTable[data & 7], 0, 88);

            out[outOffs++] = currentValue / 32768;
        }
    }

    return out;
}

/** @param {Uint8Array} wavData */
function decodeWavToSample(wavData, sampleFrequency) {
    /** @type {number[]} */
    let sampleData = [];

    let numChannels = read16LE(wavData, 22);
    let sampleRate = read32LE(wavData, 24);
    let bitsPerSample = read16LE(wavData, 34);

    console.log("decodeWav: sample rate: " + sampleRate);

    switch (bitsPerSample) {
        case 8:
        case 16:
            break;
        default:
            console.error("decodeWav: unsupported bits per sample: " + bitsPerSample);
            return;
    }

    // Number of bytes in the wav data
    let subchunk2Size = read32LE(wavData, 40);


    for (let i = 44; i < 44 + subchunk2Size; i += bitsPerSample / 8 * numChannels) {
        switch (bitsPerSample) {
            case 8:
                sampleData.push(wavData[i] / 255);
                break;
            case 16:
                sampleData.push(((read16LE(wavData, i) << 16) >> 16) / 32767);
                break;
        }
    }

    return new Sample(Float32Array.from(sampleData), sampleFrequency, sampleRate, false, 0);
}

function playStrm(strmData) {
    const BUFFER_SIZE = 4096;
    const SAMPLE_RATE = 32768;

    let bufferL = new Float32Array(BUFFER_SIZE);
    let bufferR = new Float32Array(BUFFER_SIZE);

    console.log("Number of Samples: " + read32LE(strmData, 0x24));

    let channels = strmData[0x1A];
    let numberOfBlocks = read32LE(strmData, 0x2C);
    let blockLength = read32LE(strmData, 0x30);
    let samplesPerBlock = read32LE(strmData, 0x34);
    let lastBlockLength = read32LE(strmData, 0x38);
    let lastBlockSamples = read32LE(strmData, 0x3C);

    console.log("Channels: " + channels);
    console.log("Number of blocks per channel: " + numberOfBlocks);
    console.log("Block length: " + blockLength);
    console.log("Samples per block: " + samplesPerBlock);
    console.log("Last block length: " + lastBlockLength);
    console.log("Last block samples: " + lastBlockSamples);

    if (numberOfBlocks > 2) alert("TODO: Support for block counts other than 2");
    if (channels < 2) alert("TODO: Support for mono audio");

    let sampleRate = read16LE(strmData, 0x1C);
    console.log("Sample Rate: " + sampleRate);
    console.log("Time: " + read16LE(strmData, 0x1E));

    let waveDataSize = blockLength;

    console.log("Wave data size: " + waveDataSize);

    let waveDataL = new Uint8Array(waveDataSize);
    let waveDataR = new Uint8Array(waveDataSize);

    for (let i = 0; i < waveDataSize; i++) {
        waveDataL[i] = strmData[0x68 + i];
        waveDataR[i] = strmData[0x68 + blockLength + i];
    }

    let decodedL;
    let decodedR;
    let format;
    switch (strmData[0x18]) {
        case 0: format = "PCM8"; break;
        case 1:
            format = "PCM16";
            decodedL = decodePcm16(waveDataL);
            decodedR = decodePcm16(waveDataR);
            break;
        case 2:
            format = "IMA-ADPCM";
            break;
    }

    console.log("Format: " + format);

    let inBufferPos = 0;
    let timer = 0;

    function synthesizeMore() {
        for (let i = 0; i < BUFFER_SIZE; i++) {
            bufferL[i] = decodedL[inBufferPos];
            bufferR[i] = decodedR[inBufferPos];

            timer += sampleRate;
            if (timer >= SAMPLE_RATE) {
                timer -= SAMPLE_RATE;
                if (++inBufferPos >= decodedL.length) {
                    inBufferPos = 0;
                }
            }
        }

        // console.log(inBufferPos);

        player.queueAudio(bufferL, bufferR);

        // console.log("Syntheszing more audio");
    }

    let player = new AudioPlayer(BUFFER_SIZE, SAMPLE_RATE, synthesizeMore);
    synthesizeMore();
}

/** 
 * @param {Sample} sample 
 * */
function playSample(sample) {
    return /** @type {Promise<void>} */(new Promise((resolve, reject) => {
        const BUFFER_SIZE = 4096;
        const SAMPLE_RATE = sample.sampleRate;

        let bufferL = new Float32Array(BUFFER_SIZE);
        let bufferR = new Float32Array(BUFFER_SIZE);

        let inBufferPos = 0;
        let timer = 0;

        function synthesizeMore() {

            let ended = false;

            for (let i = 0; i < BUFFER_SIZE; i++) {
                if (inBufferPos >= sample.data.length) {
                    ended = true;
                    bufferL[i] = 0;
                    bufferR[i] = 0;
                } else {
                    bufferL[i] = sample.data[inBufferPos];
                    bufferR[i] = sample.data[inBufferPos];
                }

                timer += sample.sampleRate;
                if (timer >= SAMPLE_RATE) {
                    timer -= SAMPLE_RATE;

                    inBufferPos++;
                }
            }

            if (ended) {
                resolve();
                return;
            }

            player.queueAudio(bufferL, bufferR);

        }

        let player = new AudioPlayer(BUFFER_SIZE, SAMPLE_RATE, synthesizeMore);
        synthesizeMore();
        console.log("start");
    }));
}

function midiNoteToHz(note) {
    return 440 * 2 ** ((note - 69) / 12);
}

function parseSdatFromRom(data, offset) {
    let sdat = new Sdat();

    let sdatSize = read32LE(data, offset + 0x8);
    console.log("SDAT file size: " + sdatSize);

    let sdatData = new Uint8Array(sdatSize);

    for (let i = 0; i < sdatSize; i++) {
        sdatData[i] = data[offset + i];
    }

    // downloadUint8Array("OptimePlayer extracted.sdat", sdatData);

    let numOfBlocks = read16LE(sdatData, 0xE);
    let headerSize = read16LE(sdatData, 0xC);

    console.log("Number of Blocks: " + numOfBlocks);
    console.log("Header Size: " + headerSize);

    if (headerSize > 256) {
        console.log("Header size too big (> 256), rejecting.");
        return;
    }

    let symbOffs = read32LE(sdatData, 0x10);
    let symbSize = read32LE(sdatData, 0x14);
    let infoOffs = read32LE(sdatData, 0x18);
    let infoSize = read32LE(sdatData, 0x1C);
    let fatOffs = read32LE(sdatData, 0x20);
    let fatSize = read32LE(sdatData, 0x24);
    let fileOffs = read32LE(sdatData, 0x28);
    let fileSize = read32LE(sdatData, 0x2C);

    console.log("SYMB Block Offset: " + hexN(symbOffs, 8));
    console.log("SYMB Block Size: " + hexN(symbSize, 8));
    console.log("INFO Block Offset: " + hexN(infoOffs, 8));
    console.log("INFO Block Size: " + hexN(infoSize, 8));
    console.log("FAT  Block Offset: " + hexN(fatOffs, 8));
    console.log("FAT  Block Size: " + hexN(fatSize, 8));
    console.log("FILE Block Offset: " + hexN(fileOffs, 8));
    console.log("FILE Block Size: " + hexN(fileSize, 8));

    // SYMB processing
    {
        // SSEQ symbols
        let symbSseqListOffs = read32LE(sdatData, symbOffs + 0x8);
        let symbSseqListNumEntries = read32LE(sdatData, symbOffs + symbSseqListOffs);

        console.log("SYMB Bank List Offset: " + hexN(symbSseqListOffs, 8));
        console.log("SYMB Number of SSEQ entries: " + symbSseqListNumEntries);

        for (let i = 0; i < symbSseqListNumEntries; i++) {
            let sseqNameOffs = read32LE(sdatData, symbOffs + symbSseqListOffs + 4 + i * 4);

            let sseqNameArr = [];
            let sseqNameCharOffs = 0;
            while (true) {
                let char = sdatData[symbOffs + sseqNameOffs + sseqNameCharOffs];
                if (char == 0) break; // check for null terminator
                sseqNameCharOffs++;
                sseqNameArr.push(char);
            }

            // for some reason games have a ton of empty symbols
            if (sseqNameOffs != 0) {
                let seqName = String.fromCharCode(...sseqNameArr);

                sdat.sseqNameIdDict[seqName] = i;
                sdat.sseqIdNameDict[i] = seqName;
            }
        }
    }

    {
        // SSAR symbols
        let symbSsarListOffs = read32LE(sdatData, symbOffs + 0xC);
        let symbSsarListNumEntries = read32LE(sdatData, symbOffs + symbSsarListOffs);

        console.log("SYMB Number of SSAR entries: " + symbSsarListNumEntries);
    }

    {
        // BANK symbols
        let symbBankListOffs = read32LE(sdatData, symbOffs + 0x10);
        let symbBankListNumEntries = read32LE(sdatData, symbOffs + symbBankListOffs);

        console.log("SYMB Bank List Offset: " + hexN(symbBankListOffs, 8));
        console.log("SYMB Number of BANK entries: " + symbBankListNumEntries);

        for (let i = 0; i < symbBankListNumEntries; i++) {
            let symbNameOffs = read32LE(sdatData, symbOffs + symbBankListOffs + 4 + i * 4);
            if (i == 0) console.log("NDS file addr of BANK list 1st entry: " + hexN(offset + symbOffs + symbNameOffs, 8));

            let bankNameArr = [];
            let bankNameCharOffs = 0;
            while (true) {
                let char = sdatData[symbOffs + symbNameOffs + bankNameCharOffs];
                if (char == 0) break; // check for null terminator
                bankNameCharOffs++;
                bankNameArr.push(char);
            }

            // for some reason games have a ton of empty symbols
            if (symbNameOffs != 0) {
                let bankName = String.fromCharCode(...bankNameArr);

                // console.log(bankName);
                sdat.sbnkNameIdDict[bankName] = i;
                sdat.sbnkIdNameDict[i] = bankName;
            }
        }
    }

    {
        // SWAR symbols
        let symbSwarListOffs = read32LE(sdatData, symbOffs + 0x14);
        let symbSwarListNumEntries = read32LE(sdatData, symbOffs + symbSwarListOffs);

        console.log("SYMB Number of SWAR entries: " + symbSwarListNumEntries);
    }

    // INFO processing
    {
        // SSEQ info
        let infoSseqListOffs = read32LE(sdatData, infoOffs + 0x8);
        let infoSseqListNumEntries = read32LE(sdatData, infoOffs + infoSseqListOffs);
        console.log("INFO Number of SSEQ entries: " + infoSseqListNumEntries);

        for (let i = 0; i < infoSseqListNumEntries; i++) {
            let infoSseqNameOffs = read32LE(sdatData, infoOffs + infoSseqListOffs + 4 + i * 4);

            if (infoSseqNameOffs != 0) {
                let info = new SseqInfo();
                info.fileId = read16LE(sdatData, infoOffs + infoSseqNameOffs + 0);
                info.bank = read16LE(sdatData, infoOffs + infoSseqNameOffs + 4);
                info.volume = sdatData[infoOffs + infoSseqNameOffs + 6];
                info.cpr = sdatData[infoOffs + infoSseqNameOffs + 7];
                info.ppr = sdatData[infoOffs + infoSseqNameOffs + 8];
                info.ply = sdatData[infoOffs + infoSseqNameOffs + 9];

                sdat.sseqInfos[i] = info;
                sdat.sseqList.push(i);
            } else {
                sdat.sseqInfos[i] = null;
            }
        }
    }

    {
        // SSAR info
        let infoSsarListOffs = read32LE(sdatData, infoOffs + 0xC);
        let infoSsarListNumEntries = read32LE(sdatData, infoOffs + infoSsarListOffs);
        console.log("INFO Number of SSAR entries: " + infoSsarListNumEntries);

        for (let i = 0; i < infoSsarListNumEntries; i++) {
            let infoSsarNameOffs = read32LE(sdatData, infoOffs + infoSsarListOffs + 4 + i * 4);

            if (infoSsarNameOffs != 0) {
                let info = new SsarInfo();
                info.fileId = read16LE(sdatData, infoOffs + infoSsarNameOffs + 0);

                sdat.ssarInfos[i] = info;
            } else {
                sdat.ssarInfos[i] = null;
            }
        }
    }

    {
        // BANK info
        let infoBankListOffs = read32LE(sdatData, infoOffs + 0x10);
        let infoBankListNumEntries = read32LE(sdatData, infoOffs + infoBankListOffs);
        console.log("INFO Number of BANK entries: " + infoBankListNumEntries);

        for (let i = 0; i < infoBankListNumEntries; i++) {
            let infoBankNameOffs = read32LE(sdatData, infoOffs + infoBankListOffs + 4 + i * 4);

            if (infoBankNameOffs != 0) {
                let info = new BankInfo();
                info.fileId = read16LE(sdatData, infoOffs + infoBankNameOffs + 0x0);
                info.swarId[0] = read16LE(sdatData, infoOffs + infoBankNameOffs + 0x4);
                info.swarId[1] = read16LE(sdatData, infoOffs + infoBankNameOffs + 0x6);
                info.swarId[2] = read16LE(sdatData, infoOffs + infoBankNameOffs + 0x8);
                info.swarId[3] = read16LE(sdatData, infoOffs + infoBankNameOffs + 0xA);

                sdat.sbnkInfos[i] = info;
            } else {
                sdat.sbnkInfos[i] = null;
            }
        }
    }

    {
        // SWAR info
        let infoSwarListOffs = read32LE(sdatData, infoOffs + 0x14);
        let infoSwarListNumEntries = read32LE(sdatData, infoOffs + infoSwarListOffs);
        console.log("INFO Number of SWAR entries: " + infoSwarListNumEntries);

        for (let i = 0; i < infoSwarListNumEntries; i++) {
            let infoSwarNameOffs = read32LE(sdatData, infoOffs + infoSwarListOffs + 4 + i * 4);

            if (infoSwarNameOffs) {
                let info = new SwarInfo();
                info.fileId = read16LE(sdatData, infoOffs + infoSwarNameOffs + 0x0);

                sdat.swarInfos[i] = info;
            } else {
                sdat.swarInfos[i] = null;
            }
        }
    }

    // FAT / FILE processing
    let fatNumFiles = read32LE(sdatData, fatOffs + 8);
    console.log("FAT Number of files: " + fatNumFiles);

    for (let i = 0; i < fatNumFiles; i++) {
        let fileEntryOffs = fatOffs + 0xC + i * 0x10;

        let fileDataOffs = read32LE(sdatData, fileEntryOffs);
        let fileSize = read32LE(sdatData, fileEntryOffs + 4);

        let fileData = new Uint8Array(fileSize);

        for (let j = 0; j < fileSize; j++) {
            fileData[j] = sdatData[fileDataOffs + j];
        }

        sdat.fat[i] = fileData;

        // console.log(`Loaded FAT file id:${i} size:${fileSize}`);
    }

    // Decode sound banks
    // console.log(sdat.sbnkInfos);
    for (let i = 0; i < sdat.sbnkInfos.length; i++) {
        let bank = new Bank();

        let bankInfo = sdat.sbnkInfos[i];

        if (bankInfo != null) {
            let bankFile = sdat.fat[bankInfo.fileId];

            // downloadUint8Array(sdat.sbnkIdNameDict[i] + ".sbnk", bankFile);

            let numberOfInstruments = read32LE(bankFile, 0x38);
            if (debug)
                console.log(`Bank ${i} / ${sdat.sbnkIdNameDict[i]}: ${numberOfInstruments} instruments`);
            for (let j = 0; j < numberOfInstruments; j++) {
                let fRecord = bankFile[0x3C + j * 4 + 0];
                let recordOffset = read16LE(bankFile, 0x3C + j * 4 + 1);

                let instrument = new InstrumentRecord();
                instrument.fRecord = fRecord;

                // Thanks to ipatix and pret/pokediamond
                function CalcDecayCoeff(vol) {
                    if (vol == 127)
                        return 0xFFFF;
                    else if (vol == 126)
                        return 0x3C00;
                    else if (vol < 50)
                        return (vol * 2 + 1) & 0xFFFF;
                    else
                        return (Math.floor(0x1E00 / (126 - vol))) & 0xFFFF;
                }

                // Thanks to ipatix and pret/pokediamond
                function getEffectiveAttack(attack) {
                    if (attack < 109)
                        return 255 - attack;
                    else
                        return sAttackCoeffTable[127 - attack];
                }

                // Thanks to ipatix and pret/pokediamond
                function getSustainLevel(sustain) {
                    return SNDi_DecibelSquareTable[sustain] << 7;
                }

                function readRecordData(index, offset) {
                    instrument.swavInfoId[index] = read16LE(bankFile, recordOffset + 0x0 + offset);
                    instrument.swarInfoId[index] = read16LE(bankFile, recordOffset + 0x2 + offset);
                    // if (i == 4) {
                    //     console.log(`Instrument ${j}, Record ${index}: SWAV Info ID offset:${hex(recordOffset + 0x0 + offset, 0)} value:${instrument.swavInfoId[index]}`);
                    //     console.log(`Instrument ${j}, Record ${index}: SWAR Info ID offset:${hex(recordOffset + 0x2 + offset, 0)} value:${instrument.swarInfoId[index]}`);
                    // }
                    instrument.noteNumber[index] = bankFile[recordOffset + 0x4 + offset];
                    instrument.attack[index] = bankFile[recordOffset + 0x5 + offset];
                    instrument.attackCoefficient[index] = getEffectiveAttack(instrument.attack[index]);
                    instrument.decay[index] = bankFile[recordOffset + 0x6 + offset];
                    instrument.decayCoefficient[index] = CalcDecayCoeff(instrument.decay[index]);
                    instrument.sustain[index] = bankFile[recordOffset + 0x7 + offset];
                    instrument.sustainLevel[index] = getSustainLevel(instrument.sustain[index]);
                    instrument.release[index] = bankFile[recordOffset + 0x8 + offset];
                    instrument.releaseCoefficient[index] = CalcDecayCoeff(instrument.release[index]);
                    instrument.pan[index] = bankFile[recordOffset + 0x9 + offset];
                }

                switch (fRecord) {
                    case 0: // Empty
                        break;

                    case InstrumentType.SingleSample: // Sample
                    case InstrumentType.PsgPulse: // PSG Pulse
                    case InstrumentType.PsgNoise: // PSG Noise
                        readRecordData(0, 0);
                        break;

                    case InstrumentType.Drumset: // Drumset 
                        {
                            let instrumentCount = bankFile[recordOffset + 1] - bankFile[recordOffset] + 1;

                            instrument.lowerNote = bankFile[recordOffset + 0];
                            instrument.upperNote = bankFile[recordOffset + 1];

                            for (let k = 0; k < instrumentCount; k++) {
                                readRecordData(k, 4 + k * 12);
                            }
                            break;
                        }
                    case InstrumentType.MultiSample: // Multi-Sample Instrument
                        {
                            let instrumentCount = 0;

                            for (let k = 0; k < 8; k++) {
                                let end = bankFile[recordOffset + k];
                                instrument.regionEnd[k] = end;
                                if (end == 0) {
                                    instrumentCount = k;
                                    break;
                                } else if (end == 0x7F) {
                                    instrumentCount = k + 1;
                                    break;
                                }
                            }

                            // if (i === 4) {
                            //     console.log(`Multi-Sample record offset: ${hex(recordOffset, 0)}`);
                            //     console.log(`Instrument count: ` + instrumentCount);
                            // }


                            for (let k = 0; k < instrumentCount; k++) {
                                readRecordData(k, 10 + k * 12);

                                // if (i == 4) {
                                //     console.log(`Read in record data offset:${hex(10 + k * 12, 0)}`);
                                // }

                            }
                            break;
                        }

                    default:
                        alert(`Instrument ${j}: Invalid fRecord: ${fRecord} Offset:${recordOffset}`);
                        break;
                }

                bank.instruments[j] = instrument;
            }

            sdat.banks[i] = bank;
        }
    }

    // Decode sample archives
    for (let i = 0; i < sdat.swarInfos.length; i++) {
        let archive = [];

        let swarInfo = sdat.swarInfos[i];
        if (swarInfo != null) {
            let swarFile = sdat.fat[swarInfo.fileId];

            let sampleCount = read32LE(swarFile, 0x38);
            for (let j = 0; j < sampleCount; j++) {
                let sampleOffset = read32LE(swarFile, 0x3C + j * 4);

                let wavType = swarFile[sampleOffset + 0];
                let loopFlag = swarFile[sampleOffset + 1];
                let sampleRate = read16LE(swarFile, sampleOffset + 2);
                let swarLoopOffset = read16LE(swarFile, sampleOffset + 6); // in 4-byte units
                let swarSampleLength = read32LE(swarFile, sampleOffset + 8); // in 4-byte units (excluding ADPCM header if any)

                let sampleDataLength = (swarLoopOffset + swarSampleLength) * 4;

                let sampleData = new Uint8Array(swarFile.buffer, sampleOffset + 0xC, sampleDataLength);

                let decoded;
                let loopPoint = 0;

                switch (wavType) {
                    case 0: // PCM8
                        loopPoint = swarLoopOffset * 4;
                        decoded = decodePcm8(sampleData);
                        // console.log(`Archive ${i}, Sample ${j}: PCM8`);
                        break;
                    case 1: // PCM16
                        loopPoint = swarLoopOffset * 2;
                        decoded = decodePcm16(sampleData);
                        // console.log(`Archive ${i}, Sample ${j}: PCM16`);
                        break;
                    case 2: // IMA-ADPCM
                        loopPoint = swarLoopOffset * 8 - 8;
                        decoded = decodeAdpcm(sampleData);
                        // console.log(`Archive ${i}, Sample ${j}: ADPCM`);
                        break;
                }

                if (decoded != null) {
                    archive[j] = new Sample(decoded, 440, sampleRate, loopFlag != 0, loopPoint);
                    archive[j].sampleLength = swarSampleLength * 4;
                }
            }

            sdat.sampleArchives[i] = archive;
        }
    }

    return sdat;
}

function searchForSequences(data, sequence) {
    let seqs = [];

    for (let i = 0; i < data.length; i++) {
        if (data[i] == sequence[0]) {
            for (let j = 1; j < sequence.length; j++) {
                if (data[i + j] != sequence[j]) {
                    break;
                }

                if (j == sequence.length - 1) seqs.push(i);
            }
        }
    }

    return seqs;
}

function getKeyNum(keyInOctave) {
    // THIS IS STARTING FROM THE KEY OF A
    switch (keyInOctave) {
        case 0: return 0;
        case 2: return 1;
        case 3: return 2;
        case 5: return 3;
        case 7: return 4;
        case 8: return 5;
        case 10: return 6;

        case 1: return 0;
        case 4: return 2;
        case 6: return 3;
        case 9: return 5;
        case 11: return 6;

        default: return 0;
    }
}

function isBlackKey(keyInOctave) {
    // THIS IS STARTING FROM THE KEY OF A
    switch (keyInOctave) {
        case 0: return false;
        case 2: return false;
        case 3: return false;
        case 5: return false;
        case 7: return false;
        case 8: return false;
        case 10: return false;

        case 1: return true;
        case 4: return true;
        case 6: return true;
        case 9: return true;
        case 11: return true;

        default: return false;
    }
}

const fsVisPalette = [
    "#da3fb1",
    "#ad42ba",
    "#5443c2",
    "#2b68d7",
    "#3095f2",
    "#2acdfe",
    "#2bceff",
    "#52ddf6",
    "#57d677",
    "#5ed62e",
    "#aeeb20",
    "#fef711",
    "#ff991d",
    "#ff641d",
    "#ff1434",
    "#fa30a3",
];


let activeNoteTrackNums = new Int8Array(128).fill(-1);
let lastTickTime = 0;
let lastTicks = 0;
/**
@param {CanvasRenderingContext2D} ctx 
@param {number} time 
@param {number} noteAlpha */
function drawFsVis(ctx, time, noteAlpha) {
    ctx.imageSmoothingEnabled = false;

    // normalize to 0-1 on both axes
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(ctx.canvas.width - 1, ctx.canvas.height - 1);
    ctx.fillStyle = "#222222";
    ctx.fillRect(0, 0, 1, 1);

    let wKeyWidth = 1 / 52;
    let wKeyHeight = 1 / 7;
    let pixelX = 1 / ctx.canvas.width;
    let pixelY = 1 / ctx.canvas.height;

    ctx.fillStyle = "#FF0000";
    if (currentFsVisBridge && currentBridge && currentlyPlayingSdat) {

        let activeNotes = currentFsVisBridge.activeNotes;

        if (lastTicks != currentFsVisBridge.controller.ticksElapsed) {
            lastTickTime = time;
        }
        ctx.globalAlpha = noteAlpha;

        let drew = 0;
        for (let i = 0; i < activeNotes.entries; i++) {
            let entry = activeNotes.peek(i);
            let midiNote = entry.param0;
            let duration = entry.param2;

            let bpm = currentBridge.controller.tracks[0].bpm;
            let sPerTick = (1 / (bpm / 60)) / 48;

            let ticksAdj = currentBridge.controller.ticksElapsed;
            ticksAdj += (time - lastTickTime) / 1000 / sPerTick;
            let relTime = entry.timestamp - ticksAdj;

            let pianoKey = midiNote - 21;

            let ticksToDisplay = 384;

            let height = duration / ticksToDisplay;
            let y = 1 - relTime / ticksToDisplay - height - wKeyHeight;
            if (y + height >= 1 - wKeyHeight) {
                height = 1 - wKeyHeight - y;
            }

            let octave = Math.floor(pianoKey / 12);
            let keyInOctave = pianoKey % 12;

            let keyNum = getKeyNum(keyInOctave);
            let blackKey = isBlackKey(keyInOctave);

            let whiteKeyNum = octave * 7 + keyNum;
            ctx.strokeStyle = "#444444";

            ctx.lineWidth = 0.001;

            if (y < 1 - wKeyHeight && y + height > 0) {
                if (!blackKey) {
                    ctx.fillStyle = fsVisPalette[entry.trackNum];

                    let x = whiteKeyNum * wKeyWidth;
                    let w = wKeyWidth - pixelX * 2;
                    let h = height;

                    ctx.fillRect(x, y, w, h);
                    ctx.strokeRect(x, y, w, h);

                    if (relTime < 0 && relTime > -duration) {
                        activeNoteTrackNums[midiNote] = entry.trackNum;
                    }
                } else {
                    ctx.fillStyle = fsVisPalette[entry.trackNum];

                    let x = whiteKeyNum * wKeyWidth + wKeyWidth * 0.5;
                    let w = wKeyWidth - pixelX * 2;
                    let h = height;

                    ctx.fillRect(x, y, w, h);
                    ctx.strokeRect(x, y, w, h);

                    if (relTime < 0 && relTime > -duration) {
                        activeNoteTrackNums[midiNote] = entry.trackNum;
                    }
                }
                drew++;
            }
        }

        // console.log("Drew " + drew + "Notes");

        function drawKeys(black) {
            // piano has 88 keys
            for (let j = 0; j < 88; j++) {
                let midiNote = j + 21; // lowest piano note is 21 on midi

                // using the key of A as octave base
                let octave = Math.floor(j / 12);
                let keyInOctave = j % 12;

                let keyNum = getKeyNum(keyInOctave);
                let blackKey = isBlackKey(keyInOctave);


                if (blackKey == black) {
                    let whiteKeyNum = octave * 7 + keyNum;

                    let fillStyle;
                    if (!blackKey) {
                        if (activeNoteTrackNums[midiNote] != -1) {
                            ctx.fillStyle = fsVisPalette[activeNoteTrackNums[midiNote]];
                            activeNoteTrackNums[midiNote] = -1;
                        } else {
                            ctx.fillStyle = "#ffffff";
                        }

                        let x = whiteKeyNum * wKeyWidth;
                        let y = 1 - wKeyHeight;
                        let w = wKeyWidth - pixelX * 2;
                        let h = wKeyHeight;

                        ctx.fillRect(x, y, w, h);
                    } else {
                        if (activeNoteTrackNums[midiNote] != -1) {
                            ctx.fillStyle = fsVisPalette[activeNoteTrackNums[midiNote]];
                            activeNoteTrackNums[midiNote] = -1;
                        } else {
                            ctx.fillStyle = "#000000";
                        }

                        let x = whiteKeyNum * wKeyWidth + wKeyWidth * 0.5;
                        let y = 1 - wKeyHeight;
                        let w = wKeyWidth - pixelX * 2;
                        let h = wKeyHeight * 0.58;

                        ctx.fillRect(x, y, w, h);
                    }
                }
            }
        }

        drawKeys(false);
        drawKeys(true);

        ctx.setTransform(1, 0, 0, 1, 0, 0);

        ctx.globalAlpha = 1;
        ctx.textBaseline = "top";
        ctx.fillStyle = "#ffffff";
        if (typeof process !== 'undefined') {
            // Running under node
            if (process?.env?.songName) {
                ctx.font = 'bold 48px Arial';
                ctx.fillText(`${process.env.songName}`, 24, 24);
                if (process.env.nextSongName) {
                    ctx.fillStyle = "#00ff00";
                    ctx.font = '48x Arial';
                    ctx.fillText(`Next Up: ${process.env.nextSongName}`, 24, 72);
                }
            }
        } else {
            // Running under a browser
            ctx.font = 'bold 24px monospace';
            ctx.fillText(`${currentlyPlayingSdat.sseqIdNameDict[currentlyPlayingId]} (ID: ${currentlyPlayingId})`, 24, 24);
        }
    }


    if (currentFsVisBridge)
        lastTicks = currentFsVisBridge.controller.ticksElapsed;
}