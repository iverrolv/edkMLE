clc;
clear;
store_data = true;

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

F = @(w, x) 1/N*sum(x.*exp(-1i*w*(0:1:N-1)*T));

% Data matrices:
phiData = zeros(IterK, IterSnr);
phiVar = zeros(IterK, IterSnr);
phiErrorData = zeros(IterK, IterSnr);

omegaErrorData = zeros(IterK, IterSnr);
omegaData = zeros(IterK, IterSnr);
omegaVar = zeros(IterK, IterSnr);

crlbPhiData = zeros(1, IterSnr);
crlbOmegaData = zeros(1, IterSnr);
FinEstData = zeros(IterSnr);

omegaBuffer = zeros(1, MaxIter);
phiBuffer = zeros(1, MaxIter);

finEstBuffer = zeros(1, MaxIter);
finEst = zeros(1, IterSnr);
finEstVar = zeros(1, IterSnr);
finEstError = zeros(1, IterSnr);


% create crlb:
for jsnr = 1:IterSnr
    SNR = SNRvec(jsnr);
    % White noise signal:
    tmp = 10^(SNR/10);
    sigma2 = (A^2)/(2*tmp);
    sigma = sqrt(sigma2);
    % CRLB for phase and freq:
    crlbFreq = (12*sigma2)/((A^2)*(T^2)*N*(N^2-1));
    crlbPhase =(12*sigma2*((n0^2)*N+2*n0*P+Q))/((A^2)*(N^2)*(N^2-1)); 
    crlbPhiData(jsnr) = crlbPhase;
    crlbOmegaData(jsnr) = crlbFreq;
end

for ik = 1:IterK
    M = 2^kvec(ik);
    for jsnr = 1:IterSnr
        SNR = SNRvec(jsnr);
        % White noise signal:
        tmp = 10^(SNR/10);
        sigma2 = (A^2)/(2*tmp);
        sigma = sqrt(sigma2);
        for i = 1:MaxIter
            NumI = randn(1, N);
            NumR = randn(1, N)*1i;
            wgnR = sigma*NumR;
            wgnI = sigma*NumI;
            wgn = wgnR+wgnI;
            % Create signal x[n]:
            x = A*exp(1i*(w0*n*T+phi))+wgn;
            % Frequency estimation:
            xFFT = fft(x, M);
            [~, mstar] = max(abs(xFFT));
            wFFT = 2*pi/(M*T)*mstar; 
            % Phase estimation:
            phiEst = angle(exp(-1i*wFFT*n0*T)*F(wFFT, x));
            % Store data
            omegaBuffer(i) = wFFT;
            phiBuffer(i) = phiEst;
            wFFT;
            if kvec(ik) == 10
                [wOpt, fval, exitflag, output] = fminsearch(@(w)obj(w, xFFT, x, A, n, T, F), w0);
                finEstBuffer(i)= wOpt;
                wOpt;
            end
        end
        % Store estimate for each k, for each snr:")
        phiData(ik, jsnr) = mean(phiBuffer);
        phiVar(ik, jsnr) = variance(phiBuffer, phi);
        phiErrorData(ik, jsnr) = phi - phiData(ik, jsnr);
        omegaData(ik, jsnr) = mean(omegaBuffer);
        omegaVar(ik, jsnr) = variance(omegaBuffer, w0);
        omegaErrorData(ik, jsnr) = w0 - omegaData(ik, jsnr);
        if kvec(ik) == 10
            finEst(jsnr) = mean(finEstBuffer);
            finEstError(jsnr) = w0-finEst(jsnr);
            finEstVar(jsnr) = variance(finEstBuffer, w0);
        end


    end
end
% Store data:
if store_data
    save("Data1a", "omegaData", "omegaVar", "omegaErrorData", "phiData", "phiErrorData", "phiVar", "finEstVar", "finEstError", "FinEstData");
end
%% Create plots:
%load("Data1a.mat")
SNRvec = -10:10:60;
kvec = 10:2:20;
% Plot Variance of MLE:
f1 = figure(1); clf(f1);
subplot(121);
hold on; grid on;
crlbW = log(crlbOmegaData(1, :));
% var(w)
h(1, :) = plot(SNRvec, crlbW, 'b', "LineWidth", 2);
i = 1;
for k = 1:IterK
    h(i+1) = plot(SNRvec(1:end), log(omegaVar(k, 1:end)), "LineWidth", 2);
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
h2(1, :)=plot(SNRvec, crlbPhi, 'b', "LineWidth", 2);
i=1;
for k= 1:IterK
    h2(i+1)= plot(SNRvec, log(phiVar(k, :)), "LineWidth", 2);
    i= i+1;
end
title("Variance of phase estimation error")
ylabel("log[Var($$\hat{\phi}$$)]", 'Interpreter','latex', 'FontSize', 14); 
xlabel("SNR");
legend(h2, "CRLB", "M=2^{10}", "2^{12}", "2^{14}", "2^{16}", "2^{18}", "2^{20}");


% Plot Error of MLE:
f2 = figure(2); clf;
subplot(121);
hold on; grid on;

for k=1:IterK
    h4(k) = plot(SNRvec, (omegaErrorData(k, :)), "LineWidth", 2);
end
title("Average error of $$\hat{\omega}$$ for all k", 'Interpreter', 'latex', 'FontSize',11);
ylabel("$$\omega_0-\hat{\omega}$$", 'Interpreter', 'latex', 'FontSize', 11);
xlabel("SNR");
legend(h4, "M=2^{10}", "2^{12}", "2^{14}", "2^{16}", "2^{18}", "2^{20}");

subplot(122)
hold on; grid on;
for k=1:IterK
    h3(k) = plot(SNRvec, (phiErrorData(k, :)), "LineWidth", 2);
end
title("Average error of $$\hat{\phi}$$ for all k", 'Interpreter', 'latex','FontSize',11);
ylabel("$$\phi-\hat{\phi}$$", 'Interpreter', 'latex', 'FontSize', 11);
xlabel("SNR");
legend(h3, "M=2^{10}", "2^{12}", "2^{14}", "2^{16}", "2^{18}", "2^{20}");

% Plot fine tuned estimate:
f3 = figure(3); clf(f3);
subplot(121)
hold on; grid on;
%varFinest = var(FinEstData, 0, 1);
plot(SNRvec, log(omegaVar(1,:)), 'black', 'Linewidth', 2)
plot(SNRvec, log(finEstVar), 'r', "LineWidth",2);
plot(SNRvec, (crlbW), 'b', "LineWidth", 2);
plot(SNRvec, log(omegaVar(6, :)), 'y', "LineWidth",2)
legend("k=10", "Finetuned Estimate", "CRLB", "k=20")  
subplot(122)
hold on; grid on;

plot(SNRvec, omegaErrorData(6, :), 'y', "LineWidth", 2);
plot(SNRvec, omegaErrorData(1, :), 'black', "LineWidth", 2);
plot(SNRvec, (finEstError), 'r', "LineWidth", 2)
legend( "k=20", "k=10","Finetuned Estimate")

if store_data
    exportgraphics(f1, "VarPhiOmega.eps")
    exportgraphics(f2, "MeanPhiOmega.eps")
    exportgraphics(f3, "FinEstimate.eps")
end
%% Functions
function J = obj(w, fftX, x, A, n, T, fun)
    p = angle(exp(-1i*w*n(1)*T)*fun(w, x));
    %p = pi/8;
    s = A*exp(1i*(w*n*T+p)); % Create a guess signal
    fftGuess = fft(s, 2^10);
    J = meanSquareError(abs(fftGuess), abs(fftX));
end

function mse = meanSquareError(arr1, arr2)
    mse = sum((arr1-arr2).^2)/length(arr1);
end

function vartrue = variance(data, true_mean)
    vartrue = sum((data-true_mean).^2)/length(data);
end