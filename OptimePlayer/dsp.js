// Based off NAudio's BiQuadFilter, look there for comments
class BiquadFilter {
    // coefficients
    a0 = 0;
    a1 = 0;
    a2 = 0;
    a3 = 0;
    a4 = 0;

    // state
    x1; /** @type {Float64Array} */
    x2; /** @type {Float64Array} */
    y1; /** @type {Float64Array} */
    y2; /** @type {Float64Array} */

    numCascade;

    /**
     * @param {number} order
     * @param aa0
     * @param aa1
     * @param aa2
     * @param b0
     * @param b1
     * @param b2
     */
    constructor(order, aa0 = 0, aa1 = 0, aa2 = 0, b0 = 0, b1 = 0, b2 = 0) {
        this.setCoefficients(aa0, aa1, aa2, b0, b1, b2);

        if ((order % 2) !== 0) {
            throw 'Order not divisible by 2';
        }

        this.numCascade = order / 2;

        this.x1 = new Float64Array(this.numCascade);
        this.x2 = new Float64Array(this.numCascade);
        this.y1 = new Float64Array(this.numCascade);
        this.y2 = new Float64Array(this.numCascade);
    }

    resetState() {
        for (let i = 0; i < this.numCascade; i++) {
            this.x1[i] = 0;
            this.x2[i] = 0;
            this.y1[i] = 0;
            this.y2[i] = 0;
        }
    }

    /**
     * @param {number} inSample
     */
    transform(inSample) {
        for (let i = 0; i < this.numCascade; i++) {
            // compute result
            let result = this.a0 * inSample + this.a1 * this.x1[i] + this.a2 * this.x2[i] - this.a3 * this.y1[i] - this.a4 * this.y2[i];
            // if (isNaN(result)) throw "NaN in filter";
            // if (result == Infinity) throw "Infinity in filter";

            // shift x1 to x2, sample to x1 
            this.x2[i] = this.x1[i];
            this.x1[i] = inSample;

            // shift y1 to y2, result to y1 
            this.y2[i] = this.y1[i];
            this.y1[i] = result;

            inSample = result;
        }
        return inSample;
    }

    /**
     * @param {number} aa0
     * @param {number} aa1
     * @param {number} aa2
     * @param {number} b0
     * @param {number} b1
     * @param {number} b2
     */
    setCoefficients(aa0, aa1, aa2, b0, b1, b2) {
        if (isNaN(aa0)) throw "aa0 is NaN";
        if (isNaN(aa1)) throw "aa1 is NaN";
        if (isNaN(aa2)) throw "aa2 is NaN";
        if (isNaN(b0)) throw "b0 is NaN";
        if (isNaN(b1)) throw "b1 is NaN";
        if (isNaN(b2)) throw "b2 is NaN";

        this.a0 = b0 / aa0;
        this.a1 = b1 / aa0;
        this.a2 = b2 / aa0;
        this.a3 = aa1 / aa0;
        this.a4 = aa2 / aa0;
    }

    /**
     * @param {number} sampleRate
     * @param {number} cutoffFrequency
     * @param {number} q
     */
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

    /**
     * @param {number} sampleRate
     * @param {number} centreFrequency
     * @param {number} q
     * @param {number} dbGain
     */
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

    /**
     * @param {number} sampleRate
     * @param {number} cutoffFrequency
     * @param {number} q
     */
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

    /**
     * @param {number} order
     * @param {number} sampleRate
     * @param {number} cutoffFrequency
     * @param {number} q
     */
    static lowPassFilter(order, sampleRate, cutoffFrequency, q) {
        let filter = new BiquadFilter(order);
        filter.setLowPassFilter(sampleRate, cutoffFrequency, q);
        return filter;
    }

    /**
     * @param {number} order
     * @param {number} sampleRate
     * @param {number} cutoffFrequency
     * @param {number} q
     */
    static highPassFilter(order, sampleRate, cutoffFrequency, q) {
        let filter = new BiquadFilter(order);
        filter.setHighPassFilter(sampleRate, cutoffFrequency, q);
        return filter;
    }

    /**
     * @param {number} order
     * @param {number} sampleRate
     * @param {number} centreFrequency
     * @param {number} q
     */
    static bandPassFilterConstantSkirtGain(order, sampleRate, centreFrequency, q) {
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
        return new BiquadFilter(order, a0, a1, a2, b0, b1, b2);
    }

    /**
     * @param {number} order
     * @param {number} sampleRate
     * @param {number} centreFrequency
     * @param {number} q
     */
    static bandPassFilterConstantPeakGain(order, sampleRate, centreFrequency, q) {
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
        return new BiquadFilter(order, a0, a1, a2, b0, b1, b2);
    }

    /**
     * @param {number} order
     * @param {number} sampleRate
     * @param {number} centreFrequency
     * @param {number} q
     */
    static notchFilter(order, sampleRate, centreFrequency, q) {
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
        return new BiquadFilter(order, a0, a1, a2, b0, b1, b2);
    }

    /**
     * @param {number} order
     * @param {number} sampleRate
     * @param {number} centreFrequency
     * @param {number} q
     */
    static allPassFilter(order, sampleRate, centreFrequency, q) {
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
        return new BiquadFilter(order, a0, a1, a2, b0, b1, b2);
    }

    /**
     * @param {number} order
     * @param {number} sampleRate
     * @param {number} centreFrequency
     * @param {number} q
     * @param {number} dbGain
     */
    static peakingEQ(order, sampleRate, centreFrequency, q, dbGain) {
        let filter = new BiquadFilter(order,);
        filter.setPeakingEq(sampleRate, centreFrequency, q, dbGain);
        return filter;
    }

    /**
     * @param {number} order
     * @param {number} sampleRate
     * @param {number} cutoffFrequency
     * @param {number} shelfSlope
     * @param {number} dbGain
     */
    static lowShelf(order, sampleRate, cutoffFrequency, shelfSlope, dbGain) {
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
        return new BiquadFilter(order, a0, a1, a2, b0, b1, b2);
    }

    /**
     * @param {number} order
     * @param {number} sampleRate
     * @param {number} cutoffFrequency
     * @param {number} shelfSlope
     * @param {number} dbGain
     */
    static highShelf(order, sampleRate, cutoffFrequency, shelfSlope, dbGain) {
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
        return new BiquadFilter(order, a0, a1, a2, b0, b1, b2);
    }
}
