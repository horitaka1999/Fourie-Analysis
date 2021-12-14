import numpy as np
import pyefd
class FourieMaster:
    def __init__(self,contorBox):
        self.contorBox = contorBox
    def constMatrix(self):
        zs = self.contorBox[:, 0] + (-self.contorBox[:, 1]) * 1j
        zs -= np.mean(zs)
        self.contorBox = np.array([zs.real,zs.imag]).T
        coeffs =  pyefd.elliptic_fourier_descriptors(self.contorBox,normalize = True)
        print(coeffs)
        self.zs = zs
        spx = np.fft.fft(zs.real) 
        spy = np.fft.fft(zs.imag)
        sp = np.fft.fftshift(np.fft.fft(zs))
        T = len(zs)
        self.T = T
        if T % 2 == 0:
            self.sepx = (spx[0].real + spx[T//2].real) /T
            self.sepy = (spy[0].real + spy[T//2].real) /T
        else:
            self.sepx = spx[0].real / T
            self.sepy = spy[0].real/ T
        ak = []
        bk = []
        for k in range(len(spx)):
            a = 2 * spx.real[k] / T
            b = -2 * spx.imag[k]/T
            ak.append(a)
            bk.append(b)
        ck = []
        dk = []
        for k in range(len(spy)):
            c = 2 * spy.real[k]/T
            d = -2 * spy.imag[k]/T
            ck.append(c)
            dk.append(d)
        from math import atan2,sqrt,cos,sin
        seta = (atan2(2*(ak[0] * bk[0] + ck[0] * dk[0]),(ak[0]**2+bk[0]**2 - ck[0] ** 2 - dk[0] ** 2)) + np.pi) /2
        tmp = np.array([
            [ak[0],bk[0]],
            [ck[0],dk[0]]
        ])
        tmp2 = np.dot(tmp,np.array([
            [cos(seta)],
            [sin(seta)]]))
        ap = tmp2[0]
        cp = tmp2[1]
        psy = atan2(cp,ap) +np.pi
        self.board = [
        ]#Fourie係数、標準化されている
        self.matrix = [] #Fourie係数、a,b,c,dの準
        for i in range(T):
            rev1= np.dot(np.array([
                [cos(psy),sin(psy)],
                [-sin(psy),cos(psy)]
            ]) ,
                        np.array([
                    [ak[i],bk[i]],
                            [ck[i],dk[i]]
                        ]))
            rev = np.dot(rev1,np.array([
                [cos(i*seta),-sin(i*seta)],
                [sin(i*seta),cos(i*seta)]
            ]))
            self.board.append(rev)
            if i <= 20:
                for h in range(2):
                    for w in range(2):
                        self.matrix.append(rev[h][w])
            
        E = sqrt(ap ** 2 + cp ** 2)
        self.board = np.array(self.board)/E
        return self.board

    def reconstract(self,K):
        f2s = []
        for t in range(self.T+1):
            f2Real = (self.sepx/2)
            f2Imag = (self.sepy/2)
            for k in range(K):#kは位相
                f2Real += self.board[k][0][0] * np.cos(2*np.pi * k * t / self.T) + self.board[k][0][1] * np.sin(2*np.pi*k*t/self.T)
                f2Imag += self.board[k][1][0] * np.cos(2*np.pi*k*t/self.T) + self.board[k][1][1] * np.sin(2*np.pi*k*t/self.T)
            f2s.append(f2Real + f2Imag * 1j)
        f2s = np.array(f2s)
        return f2s
