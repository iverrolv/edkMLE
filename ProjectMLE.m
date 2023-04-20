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
MaxIter = 1000;

% Data matrices:
phiData = zeros(MaxIter,IterK, IterSnr);
omegaData = zeros(MaxIter,IterK, IterSnr);
phiErrorData = zeros(MaxIter,IterK, IterSnr);
omegaErrorData = zeros(MaxIter,IterK, IterSnr);
crlbPhiData = zeros(MaxIter, IterSnr);
crlbOmegaData = zeros(MaxIter, IterSnr);
for i = 1:MaxIter
    NumI = randn(1, N);
    NumR = randn(1, N)*1i;
    for jsnr = 1:IterSnr
        SNR = SNRvec(jsnr);
        % White noise signal:
        tmp = 10^(SNR/10);
        sigma2 = (A^2)/(2*tmp);
        sigma = sqrt(sigma2);
        wgnR = sigma*NumR;
        wgnI = sigma*NumI;
        wgn = wgnR+wgnI;
        % Create signal x[n] with zero padding:
        x = A*exp(1i*(w0*n*T+phi))+wgn;
        xpad = [x zero_pad];
        % CRLB for phase and freq:
        crlbFreq = (12*sigma2)/((A^2)*(T^2)*N*(N^2-1));
        crlbPhase =(12*sigma2*((n0^2)*N+2*n0*P+Q))/((A^2)*(N^2)*(N^2-1)); 
        for jk = 1:IterK
            M = 2^kvec(jk);
            % For every k, caclulate for all SNR values.
            % Frequency estimation:
            xFFT = fft(xpad, M);

            [~, mstar] = max(abs(xFFT));
            wFFT = 2*pi/(M*T)*mstar;
            
            % Phase estimation:
            F = @(w) 1/N*sum(x.*exp(-1i*w*(0:1:N-1)*T));
            phiEst = angle(exp(-1i*wFFT*n0*T)*F(wFFT));
            % Store data
            phiData(i, jk, jsnr) = phiEst;
            phiErrorData(i, jk, jsnr) = phi - phiEst;
            omegaData(i, jk, jsnr) = wFFT;
            omegaErrorData(i, jk, jsnr) = w0 - wFFT;
            crlbPhiData(i, jsnr) = crlbPhase;
            crlbOmegaData(i, jsnr) = crlbFreq;
        end
    end
end
% Store data:
save("Data1a", "phiData", "omegaData", "crlbPhiData", "crlbOmegaData", "omegaErrorData", "phiErrorData");
%% Create plots:
load("Data1a.mat")
SNRvec = -10:10:60;
kvec = 10:2:20;
% Variance of w and phi against crlb for k and snr
f1 = figure(1); clf(f1);
subplot(121);
hold on; grid on;
crlbW = log(crlbOmegaData(1, :)); % Nå i Hz^2
varE_w = log(VarData(omegaData));
% var(w)
h(1, :) = plot(SNRvec, crlbW, 'b', "LineWidth", 2);
i = 1;
for k = 1:IterK
    h(i+1) = plot(SNRvec(1:end), varE_w(k, 1:end), "LineWidth", 2);
    i = i + 1;
end
title("Variance of frequency estimation error");
ylabel("log[Var($$\hat{\omega}$$)]", 'Interpreter','latex', 'FontSize', 14); 
xlabel("SNR");
legend(h, "CRLB", "M=2^{10}", "2^{12}", "2^{14}", "2^{16}", "2^{18}", "2^{20}")
% var(phi)
subplot(122);
hold on; grid on;
crlbPhi = log(crlbPhiData(1, :));
varPhi = log(VarData(phiErrorData));
h2(1, :)=plot(SNRvec, crlbPhi, 'b', "LineWidth", 2);
i=1;
for k= 1:IterK
    h2(i+1)= plot(SNRvec, varPhi(k, :), "LineWidth", 2);
    i= i+1;
end
title("Variance of phase estimation error")
ylabel("log[Var($$\hat{\phi}$$)]", 'Interpreter','latex', 'FontSize', 14); 
xlabel("SNR");
legend(h2, "CRLB", "M=2^{10}", "2^{12}", "2^{14}", "2^{16}", "2^{18}", "2^{20}");
exportgraphics(f1, "VarPhiOmega.eps")
%%
% e_w for k and snr.
f2 = figure(2); clf;
subplot(121);
hold on; grid on;
%meanW = MeanData(omegaData);
%meanW = w0-meanW;
meanW = MeanData(omegaErrorData)
for k=1:IterK
    h4(k) = plot(SNRvec, meanW(k, :), "LineWidth", 2);
end
title("Average error of $$\hat{\omega}$$ for all k", 'Interpreter', 'latex', 'FontSize',11);
ylabel("$$\omega_0-\hat{\omega}$$", 'Interpreter', 'latex', 'FontSize', 11);
xlabel("SNR");
legend(h4, "M=2^{10}", "2^{12}", "2^{14}", "2^{16}", "2^{18}", "2^{20}");

subplot(122)
hold on; grid on;
meanPhi = MeanData(phiErrorData);
for k=1:IterK
    h3(k) = plot(SNRvec, meanPhi(k, :), "LineWidth", 2);
end
title("Average error of $$\hat{\phi}$$ for all k", 'Interpreter', 'latex','FontSize',11);
ylabel("$$\phi-\hat{\phi}$$", 'Interpreter', 'latex', 'FontSize', 11);
xlabel("SNR");
legend(h3, "M=2^{10}", "2^{12}", "2^{14}", "2^{16}", "2^{18}", "2^{20}");
exportgraphics(f2, "MeanPhiOmega.eps")
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

function data = VarData(data)
    [~, kLen, snrLen] = size(data(1, :, :));
    out = zeros(kLen, snrLen);
    for k=1:kLen
        out(k, :) = var(data(:, k, :), 0, 1);
    end
    data = out;
end

function data = MeanData(data)
    [~, kLen, snrLen] = size(data(1, :, :));
    out = zeros(kLen, snrLen);
    for k=1:kLen
        out(k, :) = mean(data(:, k, :));
    end
    data = out;
end

