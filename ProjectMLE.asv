clc;
clear;
% Consts
Fs = 1e6;
f0 = 1e5;
T = 1e-6;
w0 = 2*pi*f0;
phi = pi/8;
A = 1;

% Samples
N = 513;
zero_padding_back = 1024-N;
zero_pad = zeros(1, zero_padding_back);
P = N*(N-1)/2;
Q = N*(N-1)*(2*N-1)/6;
n0 = -P/N; % -256
n = n0:1:n0+N-1;

SNR = 10;
SNRvec = -10:10:60;
IterSnr = length(SNRvec);
k = 20;
kvec = 10:2:20;
IterK = length(kvec);
M = 2^k;
MaxIter = 100;

% Data matrices:
phiData = zeros(MaxIter,IterK, IterSnr);
omegaData = zeros(MaxIter,IterK, IterSnr);
crlbPhiData = zeros(MaxIter, IterK, IterSnr);
crlbOmegaData = zeros(MaxIter, IterK, IterSnr);
for i = 1:MaxIter
    for jk = 1:length(kvec)
        M = 2^kvec(jk);
        for jsnr = 1:length(SNRvec)
            SNR = SNRvec(jsnr);
            % For every k, caclulate for all SNR values.
            % White noise signal:
            tmp = 10^(SNR/10);
            sigma2 = (A^2)/(2*tmp);
            sigma = sqrt(sigma2);
            wgnR = sigma*randn(1, N);
            wgnI = sigma*randn(1, N)*1i;
            wgn = wgnR+wgnI;
            % Create signal x[n] with zero padding:
            x = A*exp(1i*(w0*n*T+phi))+wgn;
            xpad = [x zero_pad];
            % CRLB for phase and freq:
            crlbFreq = (12*sigma2)/((A^2)*(T^2)*N*(N^2-1));
            crlbPhase =(12*sigma2*((n0^2)*N+2*n0*P+Q))/((A^2)*(N^2)*(N^2-1)); 
            % Frequency estimation:
            xFFT = fft(xpad, M);
            [~, mstar] = max(abs(xFFT));
            wFFT = 2*pi/(M*T)*mstar;
            % Phase estimation:
            F = @(w) 1/N*sum(x.*exp(-1i*w*(0:1:N-1)*T));
            phiEst = angle(exp(-1i*wFFT*n0*T)*F(wFFT));
            % Store data
            phiData(i, jk, jsnr) = phiEst;
            omegaData(i, jk, jsnr) = wFFT;
            crlbPhiData(i, jk, jsnr) = crlbPhase;
            crlbOmegaData(i, jk, jsnr) = crlbFreq;
        end
    end
end
% Store data:

% Create plots:


%% Fminsearch
clc;
k = 10;
M = 2^k;

% Consts
Fs = 1e6;
f0 = 1e5;
T = 1e-6;
w0 = 2*pi*f0;
phi = pi/8;
A = 1;

% Samples
N = 513;
P = N*(N-1)/2;
Q = N*(N-1)*(2*N-1)/6;
n0 = -P/N; % -256
n = n0:1:n0+N-1;

SNR = 10;

xFFT = fft(xpad, M);
absxFFT = abs(xFFT);
[mval, mstar] = max(abs(xFFT));
wFFT = 2*pi/(M*T)*mstar;
F = @(w) 1/N*sum(x.*exp(-1i*w*(0:1:N-1)*T));
phiEst = angle(exp(-1i*wFFT*n0*T)*F(wFFT));
    
wFFT
for i=0:0
    [wOpt, fval, exitflag, output] = fminsearch(@(w)fun(w, absxFFT, n0, n, N, T, A, zero_padding_back, M, x), wFFT);
    wOpt
    output
end


function mse = fun(w, absxFFT, n0, n, N, T, A, z, M, x)
    F = @(w) 1/N*sum(x.*exp(-1i*w*(0:1:N-1)*T));
    phiNew = angle(exp(-1i*w*n0*T)*F(w));
    sFFT = abs(fft([A*sin(w*n+phiNew) zeros(1, z)], M));
    error = absxFFT - sFFT;
    mse = sqrt(mean((error).^2));
end
