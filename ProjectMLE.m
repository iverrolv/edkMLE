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
P = N*(N-1)/2;
Q = N*(N-1)*(2*N-1)/6;
n0 = -P/N; % -256
n = n0:1:n0+N-1;

SNR = 10;
k = 20;
M = 2^k;
m = 0:1:M-1;
% White Gaussian noise, E=0, Var = Sigma2

Num = 100;
%
phiBar = zeros(1, Num);
for i = 1:Num
    tmp = 10^(SNR/10);
    sigma2 = (A^2)/(2*tmp);
    sigma = sqrt(sigma2);
    wgnR = sigma*randn(1, N);
    wgnI = sigma*randn(1, N)*1i;
    wgn = wgnR+wgnI;
    
    % x[n]
    x = A*exp(1i*(w0*n*T+phi))+wgn;
    zero_padding_back = 513;
    n_padded = 0:1:1023;
    xpad = [x zeros(1, zero_padding_back)];
    %plot(n, x)
    % CRLB for phase and freq:
    crlbFreq = (12*sigma2)/((A^2)*(T^2)*N*(N^2-1));
    crlbPhase =(12*sigma2*((n0^2)*N+2*n0*P+Q))/((A^2)*(N^2)*(N^2-1)); 
    
    
    xFFT = fft(xpad, M);
    [mval, mstar] = max(abs(xFFT));
    wFFT = 2*pi/(M*T)*mstar;
    
    F = @(w) 1/N*sum(x.*exp(-1i*w*(0:1:N-1)*T));
    phiEst = angle(exp(-1i*wFFT*n0*T)*F(wFFT));
    phiBar(i) = phiEst;
end

mean(phiBar)


function out = FF(w)
    tmp = 0;
    out = 1/N*tmp;

end

