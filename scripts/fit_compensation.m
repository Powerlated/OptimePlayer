function fit_compensation(dirpath)
% Measures the average spectral power loss of switching the mixer-to-output
% resampler from "Nearest neighbor" to "Sinc - output Nyquist (crunch)" on real
% Emerald DirectSound content, then fits a single biquad IIR to that loss so the
% PSG path can be coloured the same way (compensating PSGs being too loud under
% crunch).
%
% Inputs (written by `optime-cli mixer-response`):
%   near.f32   raw f32 LE, mono, nearest-neighbour mixer->output
%   crunch.f32 raw f32 LE, mono, output-Nyquist crunch mixer->output
%   meta.txt   key=value lines (out_rate, ...)

if nargin < 1, dirpath = '.'; end
fs = readmeta(fullfile(dirpath,'meta.txt'), 'out_rate', 48000);

xn = readf32(fullfile(dirpath,'near.f32'));
xc = readf32(fullfile(dirpath,'crunch.f32'));
n  = min(numel(xn), numel(xc));
xn = xn(1:n); xc = xc(1:n);

% ---- Welch PSDs on the same grid ----------------------------------------
nfft = 8192;
win  = hann(nfft);
nov  = nfft/2;
[Pn,f] = pwelch(xn, win, nov, nfft, fs);
[Pc,~] = pwelch(xc, win, nov, nfft, fs);

% Guard against silent bins.
flo = max(Pn) * 1e-8;
Pn  = max(Pn, flo);
Pc  = max(Pc, flo);

ratio   = Pc ./ Pn;            % power transfer (crunch / nearest)
ratiodB = 10*log10(ratio);
mag     = sqrt(ratio);         % magnitude transfer |H(f)|

% ---- Average spectral power loss ----------------------------------------
% Content-weighted: total crunch power / total nearest power across the band
% (i.e. weighted by where the real DirectSound spectrum actually has energy).
avg_pow_ratio = sum(Pc) / sum(Pn);
avg_loss_dB   = 10*log10(avg_pow_ratio);
% Unweighted mean of the per-bin loss, for reference.
mean_bin_dB   = mean(ratiodB);

fprintf('\n=== Average spectral power loss (crunch vs nearest) ===\n');
fprintf('content-weighted total-power ratio : %.4f  (%.3f dB)\n', avg_pow_ratio, avg_loss_dB);
fprintf('unweighted mean per-bin loss       : %.3f dB\n', mean_bin_dB);

% ---- Fit a PER-RATE RBJ low-pass (cutoff in Hz, Q) to the rolloff ---------
% The engine realizes the compensation as `BiquadFilter::low_pass(order, fs, fc, Q)`
% -- a cascade of `order/2` IDENTICAL RBJ low-pass sections -- so the filter is
% rebuilt at whatever output rate the device runs at (the knee stays at a fixed
% fc in Hz instead of a baked, rate-specific z-domain shape). We therefore fit the
% rate-independent parameters (fc, Q) rather than z coefficients, and sweep the
% cascade order to answer "do more stages help?".
%
% Fit band: the audible knee/transition where PSG & DirectSound energy actually
% lives; a biquad cascade cannot reach the crunch's near-brick-wall deep stopband,
% so weighting that region only fights the passband. Clamp the target floor too.
w        = pi * f / (fs/2);               % digital frequency [0, pi]
% Fit the audible knee/transition. The crunch's near-brick-wall stopband above
% ~16.5 kHz is unreachable by a low-order cascade and perceptually empty on real
% content, so including it just drags the cutoff down without audible benefit.
fit_band = f <= 16500;
% The compensation only ever DARKENS the PSG bus to match DirectSound, so we model
% just the attenuation: smooth the noisy Welch ratio and clamp it to <= 0 dB (the
% slight >0 dB ZOH-droop region would otherwise ask the filter to boost PSG, the
% wrong direction), with a floor at the deepest reach a biquad cascade can track.
mag_sm   = movmean(mag, 9);
tgt_dB   = min(20*log10(mag_sm), 0);
tgt_dB   = max(tgt_dB, -24);

orders = [2 4 6 8];
fprintf('\n=== Per-rate RBJ low-pass fits (identical cascade) ===\n');
best = struct('rms', inf);
opt = optimset('Display','off','TolX',1e-4,'TolFun',1e-6);
cost = @(p, od) rbj_cost(p, od, fs, w, tgt_dB, fit_band);
for od = orders
    p0 = [13000, 0.7];
    p  = fminsearch(@(p) cost(p, od), p0, opt);
    rms = sqrt(cost(p, od));
    fprintf('order %d: fc=%7.1f Hz  Q=%.4f  band-RMS=%.3f dB', od, p(1), p(2), rms);
    g15 = rbj_lp_db(p(1), p(2), od, fs, w);
    pick = @(hz) g15(find(f>=hz,1));
    fprintf('   [10k:%+.1f 13k:%+.1f 15k:%+.1f 18k:%+.1f dB]\n', ...
            pick(10000), pick(13000), pick(15000), pick(18000));
    if rms < best.rms
        best = struct('rms',rms,'order',od,'fc',p(1),'Q',p(2));
    end
end

% Reference upper bound: general (non-identical) yulewalk designs of the same
% order, to judge how much accuracy a more general cascade *could* buy.
fprintf('\n=== Reference: general yulewalk fits (non-identical sections) ===\n');
fn = f/(fs/2); mclamp = min(mag,1.2);
for od = orders
    [by,ay] = yulewalk(od, fn(:)', mclamp(:)');
    Hy = freqz(by, ay, w);
    dy = 20*log10(abs(Hy)) - tgt_dB;
    fprintf('order %d: band-RMS=%.3f dB (general IIR upper bound)\n', od, sqrt(mean(dy(fit_band).^2)));
end

% Pick the smallest order within 0.25 dB of the best RMS (diminishing returns).
sel = best;
for od = orders
    p  = fminsearch(@(p) cost(p, od), [13000,0.7], opt);
    if sqrt(cost(p,od)) <= best.rms + 0.25
        sel = struct('rms',sqrt(cost(p,od)),'order',od,'fc',p(1),'Q',p(2));
        break;
    end
end
fprintf('\nSELECTED: order %d, fc=%.1f Hz, Q=%.4f (band-RMS %.3f dB)\n', ...
        sel.order, sel.fc, sel.Q, sel.rms);

% Realized content-weighted loss with the selected filter.
Hsel  = 10.^(rbj_lp_db(sel.fc, sel.Q, sel.order, fs, w)/20);
selloss_dB = 10*log10(sum((Hsel.^2).*Pn)/sum(Pn));
fprintf('selected filter content-weighted loss: %.3f dB (measured %.3f dB)\n', selloss_dB, avg_loss_dB);

% ---- Emit Rust-ready parameters -----------------------------------------
fid = fopen(fullfile(dirpath,'coeffs.txt'),'w');
fprintf(fid, '// mixer-to-output crunch PSG compensation: per-rate RBJ low-pass\n');
fprintf(fid, '// fit on real Emerald DirectSound; built per output rate via BiquadFilter::low_pass\n');
fprintf(fid, '// measured content-weighted loss %.3f dB; band-RMS fit err %.3f dB\n', avg_loss_dB, sel.rms);
fprintf(fid, 'order=%d\nfc_hz=%.4f\nq=%.6f\n', sel.order, sel.fc, sel.Q);
fclose(fid);
fprintf('wrote %s\n', fullfile(dirpath,'coeffs.txt'));

% ---- Plot ---------------------------------------------------------------
fig = figure('Visible','off','Position',[100 100 900 600]);
semilogx(f, ratiodB, 'b', 'LineWidth', 1); hold on;
for od = orders
    p = fminsearch(@(p) cost(p, od), [13000,0.7], opt);
    semilogx(f, rbj_lp_db(p(1),p(2),od,fs,w), '--', 'LineWidth', 1.2, ...
             'DisplayName', sprintf('RBJ LP order %d', od));
end
yline(avg_loss_dB, 'k:', sprintf('avg %.2f dB', avg_loss_dB));
grid on; xlim([20 fs/2]); ylim([-24 6]);
xlabel('Frequency (Hz)'); ylabel('Power transfer crunch/nearest (dB)');
legend('measured (Welch ratio)','Location','southwest');
title('Mixer-to-output nearest\rightarrowcrunch loss: per-rate RBJ LP fits by order');
saveas(fig, fullfile(dirpath,'compensation_fit.png'));
fprintf('wrote %s\n', fullfile(dirpath,'compensation_fit.png'));
end

function mdb = rbj_lp_db(fc, Q, order, fs, w)
% Magnitude (dB) of an `order`-section identical-cascade RBJ low-pass at digital
% frequencies w, matching `BiquadFilter::low_pass(order, fs, fc, Q)` in the engine.
    w0 = 2*pi*fc/fs; cw = cos(w0); al = sin(w0)/(2*Q);
    b0 = (1-cw)/2; b1 = 1-cw; b2 = (1-cw)/2;
    a0 = 1+al;     a1 = -2*cw; a2 = 1-al;
    H1 = zresp([b0 b1 b2], [a0 a1 a2], w);   % zresp divides num/den, a0 cancels
    mdb = (order/2) * 20*log10(abs(H1));
end

function e = rbj_cost(p, order, fs, w, tgt_dB, fit_band)
    fc = p(1); Q = p(2);
    % Q capped at 0.707 (no resonant boost -- the filter must only attenuate).
    if fc <= 6000 || fc >= fs/2 || Q < 0.5 || Q > 0.707, e = 1e9; return; end
    d = rbj_lp_db(fc, Q, order, fs, w) - tgt_dB;
    e = mean(d(fit_band).^2);
end

function H = zresp(b, a, w)
% Frequency response of b(z)/a(z) at digital frequency w (rad/sample), z=e^{jw}.
    z1 = exp(-1j*w); z2 = exp(-2j*w);
    H = (b(1) + b(2).*z1 + b(3).*z2) ./ (a(1) + a(2).*z1 + a(3).*z2);
end

function v = readf32(p)
    fid = fopen(p,'r'); v = fread(fid, inf, 'float32=>double'); fclose(fid);
end

function v = readmeta(p, key, dflt)
    v = dflt;
    txt = fileread(p);
    m = regexp(txt, [key '=([0-9eE\.\+\-]+)'], 'tokens', 'once');
    if ~isempty(m), v = str2double(m{1}); end
end
