'''
Warping Function for Turbulence Simulator

Python version
of original code by Stanley Chan

Zhiyuan Mao and Stanley Chan
Copyright 2021
Purdue University, West Lafayette, In, USA.
'''

import numpy as np
from skimage.transform import resize

try:
	import torch
	import torch.nn.functional as F
	_TORCH_AVAILABLE = True
except Exception:
	torch = None
	F = None
	_TORCH_AVAILABLE = False


def _resize_torch_2d(arr: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
	arr4 = arr.unsqueeze(0).unsqueeze(0)
	out = F.interpolate(arr4, size=(int(out_h), int(out_w)), mode='bilinear', align_corners=False)
	return out.squeeze(0).squeeze(0)


def _motion_compensate_cpu(img, Mvx, Mvy, pel):
	m, n = np.shape(img)[0], np.shape(img)[1]
	img = resize(img, (np.int32(m / pel), np.int32(n / pel)), mode='reflect')
	Blocksize = np.floor(np.shape(img)[0] / np.shape(Mvx)[0])
	m, n = np.shape(img)[0], np.shape(img)[1]
	M, N = np.int32(np.ceil(m / Blocksize) * Blocksize), np.int32(np.ceil(n / Blocksize) * Blocksize)

	f = img[0:M, 0:N]

	Mvxmap = resize(Mvy, (N, M))
	Mvymap = resize(Mvx, (N, M))

	xgrid, ygrid = np.meshgrid(np.arange(0, N - 0.99), np.arange(0, M - 0.99))
	X = np.clip(xgrid + np.round(Mvxmap / pel), 0, N - 1)
	Y = np.clip(ygrid + np.round(Mvymap / pel), 0, M - 1)

	idx = np.int32(Y.flatten() * N + X.flatten())
	f_vec = f.flatten()
	g = np.reshape(f_vec[idx], [N, M])

	g = resize(g, (np.shape(g)[0] * pel, np.shape(g)[1] * pel))
	return g


def motion_compensate(img, Mvx, Mvy, pel, device=None):
	use_cuda = (
		_TORCH_AVAILABLE
		and torch.cuda.is_available()
		and str(device).lower().startswith('cuda')
	)
	if not use_cuda:
		return _motion_compensate_cpu(img=img, Mvx=Mvx, Mvy=Mvy, pel=pel)

	m, n = np.shape(img)[0], np.shape(img)[1]
	out_h = int(np.int32(m / pel))
	out_w = int(np.int32(n / pel))

	img_t = torch.as_tensor(img, dtype=torch.float32, device=device)
	img_t = _resize_torch_2d(img_t, out_h=out_h, out_w=out_w)

	block_size = float(np.floor(float(img_t.shape[0]) / float(np.shape(Mvx)[0])))
	if block_size <= 0:
		return _motion_compensate_cpu(img=img, Mvx=Mvx, Mvy=Mvy, pel=pel)

	m2, n2 = int(img_t.shape[0]), int(img_t.shape[1])
	M = int(np.int32(np.ceil(m2 / block_size) * block_size))
	N = int(np.int32(np.ceil(n2 / block_size) * block_size))
	f = img_t[0:M, 0:N]

	Mvx_t = torch.as_tensor(Mvx, dtype=torch.float32, device=device)
	Mvy_t = torch.as_tensor(Mvy, dtype=torch.float32, device=device)
	Mvxmap = _resize_torch_2d(Mvy_t, out_h=N, out_w=M)
	Mvymap = _resize_torch_2d(Mvx_t, out_h=N, out_w=M)

	xgrid = torch.arange(0, N, dtype=torch.float32, device=device).unsqueeze(0).repeat(M, 1)
	ygrid = torch.arange(0, M, dtype=torch.float32, device=device).unsqueeze(1).repeat(1, N)
	X = torch.clamp(xgrid + torch.round(Mvxmap / float(pel)), min=0.0, max=float(N - 1))
	Y = torch.clamp(ygrid + torch.round(Mvymap / float(pel)), min=0.0, max=float(M - 1))

	idx = (Y.reshape(-1).to(dtype=torch.int64) * int(N) + X.reshape(-1).to(dtype=torch.int64))
	f_vec = f.reshape(-1)
	g = f_vec[idx].reshape(N, M)

	g = _resize_torch_2d(g, out_h=int(np.shape(g)[0] * pel), out_w=int(np.shape(g)[1] * pel))
	return g.detach().cpu().numpy()