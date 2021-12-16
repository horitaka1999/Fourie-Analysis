from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np

try:
    _range = xrange
except NameError:
    _range = range
class FourieMaster:
    def __init__(self,contorBox):
        self.contorBox = contorBox
        zs = self.contorBox[:, 0] + (-self.contorBox[:, 1]) * 1j
        zs -= np.mean(zs)
        self.contor = np.array([zs.real,zs.imag]).T


    def elliptic_fourier_descriptors(self,
        contour, order=10, normalize=False, return_transformation=False
    ):
        """Calculate elliptical Fourier descriptors for a contour.

        :param numpy.ndarray contour: A contour array of size ``[M x 2]``.
        :param int order: The order of Fourier coefficients to calculate.
        :param bool normalize: If the coefficients should be normalized;
            see references for details.
        :param bool return_transformation: If the normalization parametres should be returned.
            Default is ``False``.
        :return: A ``[order x 4]`` array of Fourier coefficients and optionally the
            transformation parametres ``scale``, ``psi_1`` (rotation) and ``theta_1`` (phase)
        :rtype: ::py:class:`numpy.ndarray` or (:py:class:`numpy.ndarray`, (float, float, float))

        """
        dxy = np.diff(contour, axis=0)
        dt = np.sqrt((dxy ** 2).sum(axis=1))
        t = np.concatenate([([0.0]), np.cumsum(dt)])
        T = t[-1]

        phi = (2 * np.pi * t) / T

        orders = np.arange(1, order + 1)
        consts = T / (2 * orders * orders * np.pi * np.pi)
        phi = phi * orders.reshape((order, -1))

        d_cos_phi = np.cos(phi[:, 1:]) - np.cos(phi[:, :-1])
        d_sin_phi = np.sin(phi[:, 1:]) - np.sin(phi[:, :-1])

        a = consts * np.sum((dxy[:, 0] / dt) * d_cos_phi, axis=1)
        b = consts * np.sum((dxy[:, 0] / dt) * d_sin_phi, axis=1)
        c = consts * np.sum((dxy[:, 1] / dt) * d_cos_phi, axis=1)
        d = consts * np.sum((dxy[:, 1] / dt) * d_sin_phi, axis=1)

        coeffs = np.concatenate(
            [
                a.reshape((order, 1)),
                b.reshape((order, 1)),
                c.reshape((order, 1)),
                d.reshape((order, 1)),
            ],
            axis=1,
        )

        if normalize:
            coeffs = self.normalize_efd(coeffs, return_transformation=return_transformation)

        return coeffs


    def normalize_efd(self,coeffs, size_invariant=True, return_transformation=False):
        """Normalizes an array of Fourier coefficients.

        See [#a]_ and [#b]_ for details.

        :param numpy.ndarray coeffs: A ``[n x 4]`` Fourier coefficient array.
        :param bool size_invariant: If size invariance normalizing should be done as well.
            Default is ``True``.
        :param bool return_transformation: If the normalization parametres should be returned.
            Default is ``False``.
        :return: The normalized ``[n x 4]`` Fourier coefficient array and optionally the
            transformation parametres ``scale``, :math:`psi_1` (rotation) and :math:`theta_1` (phase)
        :rtype: :py:class:`numpy.ndarray` or (:py:class:`numpy.ndarray`, (float, float, float))

        """
        # Make the coefficients have a zero phase shift from
        # the first major axis. Theta_1 is that shift angle.
        theta_1 = 0.5 * np.arctan2(
            2 * ((coeffs[0, 0] * coeffs[0, 1]) + (coeffs[0, 2] * coeffs[0, 3])),
            (
                (coeffs[0, 0] ** 2)
                - (coeffs[0, 1] ** 2)
                + (coeffs[0, 2] ** 2)
                - (coeffs[0, 3] ** 2)
            ),
        )
        # Rotate all coefficients by theta_1.
        for n in _range(1, coeffs.shape[0] + 1):
            coeffs[n - 1, :] = np.dot(
                np.array(
                    [
                        [coeffs[n - 1, 0], coeffs[n - 1, 1]],
                        [coeffs[n - 1, 2], coeffs[n - 1, 3]],
                    ]
                ),
                np.array(
                    [
                        [np.cos(n * theta_1), -np.sin(n * theta_1)],
                        [np.sin(n * theta_1), np.cos(n * theta_1)],
                    ]
                ),
            ).flatten()

        # Make the coefficients rotation invariant by rotating so that
        # the semi-major axis is parallel to the x-axis.
        psi_1 = np.arctan2(coeffs[0, 2], coeffs[0, 0])
        psi_rotation_matrix = np.array(
            [[np.cos(psi_1), np.sin(psi_1)], [-np.sin(psi_1), np.cos(psi_1)]]
        )
        # Rotate all coefficients by -psi_1.
        for n in _range(1, coeffs.shape[0] + 1):
            coeffs[n - 1, :] = psi_rotation_matrix.dot(
                np.array(
                    [
                        [coeffs[n - 1, 0], coeffs[n - 1, 1]],
                        [coeffs[n - 1, 2], coeffs[n - 1, 3]],
                    ]
                )
            ).flatten()

        size = coeffs[0, 0]
        if size_invariant:
            # Obtain size-invariance by normalizing.
            coeffs /= np.abs(size)

        if return_transformation:
            return coeffs, (size, psi_1, theta_1)
        else:
            return coeffs


    def calculate_dc_coefficients(self,contour):
        """Calculate the :math:`A_0` and :math:`C_0` coefficients of the elliptic Fourier series.

        :param numpy.ndarray contour: A contour array of size ``[M x 2]``.
        :return: The :math:`A_0` and :math:`C_0` coefficients.
        :rtype: tuple

        """
        dxy = np.diff(contour, axis=0)
        dt = np.sqrt((dxy ** 2).sum(axis=1))
        t = np.concatenate([([0.0]), np.cumsum(dt)])
        T = t[-1]

        xi = np.cumsum(dxy[:, 0]) - (dxy[:, 0] / dt) * t[1:]
        A0 = (1 / T) * np.sum(((dxy[:, 0] / (2 * dt)) * np.diff(t ** 2)) + xi * dt)
        delta = np.cumsum(dxy[:, 1]) - (dxy[:, 1] / dt) * t[1:]
        C0 = (1 / T) * np.sum(((dxy[:, 1] / (2 * dt)) * np.diff(t ** 2)) + delta * dt)

        # A0 and CO relate to the first point of the contour array as origin.
        # Adding those values to the coefficients to make them relate to true origin.
        return A0 ,  C0 


    def reconstruct_contour(self,coeffs, locus=(0, 0), num_points=300):
        """Returns the contour specified by the coefficients.

        :param coeffs: A ``[n x 4]`` Fourier coefficient array.
        :type coeffs: numpy.ndarray
        :param locus: The :math:`A_0` and :math:`C_0` elliptic locus in [#a]_ and [#b]_.
        :type locus: list, tuple or numpy.ndarray
        :param num_points: The number of sample points used for reconstructing the contour from the EFD.
        :type num_points: int
        :return: A list of x,y coordinates for the reconstructed contour.
        :rtype: numpy.ndarray

        """
        t = np.linspace(0, 1.0, num_points)
        # Append extra dimension to enable element-wise broadcasted multiplication
        coeffs = coeffs.reshape(coeffs.shape[0], coeffs.shape[1], 1)

        orders = coeffs.shape[0]
        orders = np.arange(1, orders + 1).reshape(-1, 1)
        order_phases = 2 * orders * np.pi * t.reshape(1, -1)

        xt_all = coeffs[:, 0] * np.cos(order_phases) + coeffs[:, 1] * np.sin(order_phases)
        yt_all = coeffs[:, 2] * np.cos(order_phases) + coeffs[:, 3] * np.sin(order_phases)

        xt_all = xt_all.sum(axis=0)
        yt_all = yt_all.sum(axis=0)
        xt_all = xt_all + np.ones((num_points,)) * locus[0]
        yt_all = yt_all + np.ones((num_points,)) * locus[1]

        reconstruction = np.stack([xt_all, yt_all], axis=1)
        return reconstruction


    def plot_efd(self,coeffs, locus=(0.0, 0.0), image=None, contour=None, n=300,K = 20):
        """Plot a ``[2 x (N / 2)]`` grid of successive truncations of the series.

        .. note::

            Requires `matplotlib <http://matplotlib.org/>`_!

        :param numpy.ndarray coeffs: ``[N x 4]`` Fourier coefficient array.
        :param list, tuple or numpy.ndarray locus:
            The :math:`A_0` and :math:`C_0` elliptic locus in [#a]_ and [#b]_.
        :param int n: Number of points to use for plotting of Fourier series.

        """

        N = coeffs.shape[0]
        N_half = int(np.ceil(N / 2))
        n_rows = 2

        t = np.linspace(0, 1.0, n)
        xt = np.ones((n,)) * locus[0]
        yt = np.ones((n,)) * locus[1]

        for k in _range(K):
            xt += (coeffs[k, 0] * np.cos(2 * (k + 1) * np.pi * t)) + (
                coeffs[k, 1] * np.sin(2 * (k + 1) * np.pi * t)
            )
            yt += (coeffs[k, 2] * np.cos(2 * (k + 1) * np.pi * t)) + (
                coeffs[k, 3] * np.sin(2 * (k + 1) * np.pi * t)
            )
        return (xt,yt)
    def constMatrix(self,order_num = 100):
        self.coeffs = self.elliptic_fourier_descriptors(self.contor,order=order_num,normalize= True)
        self.a0, self.c0 = self.calculate_dc_coefficients(self.contor)
        self.matrix = self.coeffs
    def rotateMatrix(self,seta):
        rev = np.array([
            [np.cos(seta),-np.sin(seta)],[np.sin(seta),np.cos(seta)]
        ])
        return rev

    def reconstract(self,Knum):
        if Knum >= self.coeffs.shape[0]: 
            Knum = self.coeffs.shape[0] 
        xt,yt = self.plot_efd(self.coeffs,locus=(self.a0,self.c0),K = Knum)
        psi= np.arctan2(self.contor[0,1] - self.c0, self.contor[0,0] -self.a0)
        rotate = self.rotateMatrix(psi)
        newx = []
        newy = []
        mirror = np.array([
            [1,0],[0,-1]
        ])
        for i in range(len(xt)):
            tmp = np.dot(mirror,np.dot(rotate,np.array([xt[i],yt[i]]).T))
            x = tmp[0]
            y = tmp[1]
            newx.append(x)
            newy.append(y)
        newx = np.array(newx)
        newy = np.array(newy)

        rev = newx + newy * 1j
        return rev


    def calcPCA(self):
        K = 4
        rev = []
        dim = 30
        data = self.matrix.flatten()[3:4*dim]#fourier coeffs
        eigenVector = np.array(np.load('eigenVector.npy'))
        for i in range(K):
            rev.append(calcScore(eigenVector,i,data))
        return rev




        
def calcScore(W,k,coeffs):#calc PCA score using kth eigen vector
    return np.dot(W[:,k],coeffs)