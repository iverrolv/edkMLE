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
MaxIter = 10;

% Data matrices:
phiData = zeros(MaxIter,IterK, IterSnr);
omegaData = zeros(MaxIter,IterK, IterSnr);
crlbPhiData = zeros(MaxIter, IterSnr);
crlbOmegaData = zeros(MaxIter, IterSnr);
for i = 1:MaxIter
    for jk = 1:IterK
        M = 2^kvec(jk);
        for jsnr = 1:IterSnr
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
            crlbPhiData(i, jsnr) = crlbPhase;
            crlbOmegaData(i, jsnr) = crlbFreq;
        end
    end
end
% Store data:
save("Data1a", "phiData", "omegaData", "crlbPhiData", "crlbOmegaData");
%save("Data1aAscii", "phiData", "omegaData", "crlbPhiData","crlbOmegaData", "-ascii", "-double", "-tabs"); 
%Must convert to 2D array if textfile
%% Create plots:
load("Data1a.mat")
SNRvec = -10:10:60;
kvec = 10:2:20;
%1)
% Variasn til wFFT mot CRLB for k og snr
f1 = figure(1); clf(f1);
title("Variance for ")
subplot(121);
hold on; grid on;
crlbW = log((crlbOmegaData./(4*(pi^2)))); % Nå i Hz^2
% ønsker var av freqError
varW = VarData(omegaData);
plot(SNRvec, crlbW, 'b', "LineWidth", 2);
i = 1;
cleanVecs = [1, 1, 3, 5, 6, 7];  % mh...
for k = 1:IterK
    plot(SNRvec(1:cleanVecs(i)), log(varW(k, 1:cleanVecs(i))), "LineWidth", 2);
    i = i + 1;
end
legend("CRLB", "M=2^{10}", "2^{12}", "2^{14}", "2^{16}", "2^{18}", "2^{20}")
% Varians til phiEst mot CRLB for k og snr
subplot(122);
hold on; grid on;
crlbPhi = (MeanCRLB(crlbPhiData));
varPhi = VarData(phiData);
plot(SNRvec, log(crlbPhi), 'b', "LineWidth", 2);
for k= 1:IterK
    plot(SNRvec, log(varPhi(k, :)), "LineWidth", 2);
end
legend("CRLB", "M=2^{10}", "2^{12}", "2^{14}", "2^{16}", "2^{18}", "2^{20}");
exportgraphics(f1, "VarPhiOmega.eps")

% 2)
% Gjennosnittlig frekvesnavvik wFFT mot w0 for k og snr
% Gjnsnlig faseavvik phiEst mot phi for k og snr



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

function crlb = MeanCRLB(crlbData)
    crlb = var(crlbData);
end

function data = VarData(data)
    [~, kLen, snrLen] = size(data(1, :, :));
    out = zeros(kLen, snrLen);
    for k=1:kLen
        out(k, :) = var(data(:, k, :));
    end
    data = out;
end