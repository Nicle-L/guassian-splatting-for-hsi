from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum import rasterize_gaussians_sum
from utils import *
import torch
import torch.nn as nn
import numpy as np
import math
from quantize import *
from optimizer import Adan

class GaussianImage_Cholesky(nn.Module):
    def __init__(self, loss_type="L2", **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.init_num_points = kwargs["num_points"] #800 1000 3000 5000 7000 9000
        self.iterations=kwargs["iterations"]
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]#图像分块 加速训练
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        ) #一共分为几块

        self.device = kwargs["device"]
        self.add_rate = 0.2
        self.rgb_W = nn.Parameter(torch.ones(self.init_num_points, 1))
        self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)))# 每个点有两个值（x, y），初始值在 [-1, 1]。看做点采样
        self._cholesky = nn.Parameter(torch.rand(self.init_num_points, 3))#每个高斯点的协方差矩阵的 Cholesky 分解元素，下三角矩阵因此是3
        self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))#透明度
        self._features_dc = nn.Parameter(torch.rand(self.init_num_points, 3))#每个点的RGB颜色
        self.last_size = (self.H, self.W) #原始的H和W
        self.quantize = kwargs["quantize"]
        self.register_buffer('background', torch.ones(3))
        self.opacity_activation = torch.sigmoid
        self.rgb_activation = torch.sigmoid
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))#它代表了在渲染时用于限制坐标范围的“边界偏移”，常用于高斯点的位置 x, y 的范围控制。
        self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 3))#它用于满足协方差矩阵的正定性

        if self.quantize:
            self.xyz_quantizer = FakeQuantizationHalf.apply #将数据转为半精度浮点数float16，在转化位float32.量化感知训练
            self.features_dc_quantizer = VectorQuantizer(codebook_dim=3, codebook_size=8, num_quantizers=2, vector_type="vector", kmeans_iters=5)
            self.cholesky_quantizer = UniformQuantizer(signed=False, bits=6, learned=True, num_channels=3) # 均匀量化 cholesk矩阵压缩

        self.lr = kwargs["lr"]
        self.opt_type =kwargs["opt_type"]
        if kwargs["opt_type"] == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=kwargs["lr"])
        else:
            self.optimizer = Adan(self.parameters(), lr=kwargs["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)

    def _init_data(self):
        self.cholesky_quantizer._init_data(self._cholesky)

    @property
    def get_xyz(self):
        return torch.tanh(self._xyz)#(-1,1)
    
    @property
    def get_features(self):
        return self._features_dc*self.get_rgb_W
    
    @property
    def get_rgb_W(self):
        return self.rgb_W
    @property
    def get_opacity(self):
        return self._opacity

    @property
    def get_cholesky_elements(self):
        return self._cholesky+self.cholesky_bound

    def forward(self):
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d (self.get_xyz, self.get_cholesky_elements, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                self.get_features, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)#background没有进行高斯渲染时选择白色背景
        out_img = torch.clamp(out_img, 0, 1) 
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()     

        return {"render": out_img}
    
    def update_optimizer(self):
        if self.opt_type == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            self.optimizer = Adan(self.parameters(), lr=self.lr)
            
    def train_use_iter(self, gt_image, iter, xyz_coordinates=None, rgb_values=None, rgb_weight=None):
        # If it's the first iteration and no xyz_coordinates or rgb_values are given
        if iter == 1:
            if xyz_coordinates is None or rgb_values is None:
                # First iteration, no provided points, so we take random points from the forward pass
                render_pkg1 = self.forward()
            else:

                xyz_coordinates = torch.tensor(xyz_coordinates).to(self._xyz.device)
                rgb_values = torch.tensor(rgb_values).to(self._features_dc.device)
                rgb_weight= torch.tensor(rgb_weight).to(self.rgb_W.device)
                
                rgb_weight = torch.norm(self.rgb_W, dim=1)
                _, sorted_indices = torch.sort(rgb_weight, descending=True)
                top_20_percent = int(0.2 * self.init_num_points)
                remain_indices = sorted_indices[:top_20_percent]
                keep_indices = torch.ones(self._xyz.shape[0], dtype=torch.bool, device=self._xyz.device)
                keep_indices[remain_indices] = True
                
                
                self._xyz = nn.Parameter(self._xyz[remain_indices])
                self._cholesky = nn.Parameter(self._cholesky[remain_indices])
                self._features_dc = nn.Parameter(self._features_dc[remain_indices])
                self.rgb_W = torch.nn.Parameter(self.rgb_W[remain_indices])
    
                remain_num = int(self.init_num_points * 0.8)
                
                new_xyz = torch.atanh(2 * (torch.rand(remain_num, 2) - 0.5)).to(self._xyz.device)
                new_cholesky = torch.rand(remain_num, 3).to(self._xyz.device)
                new_features_dc = torch.rand(remain_num, 3).to(self._xyz.device)
                new_rgb_W = 0.01 * torch.ones(remain_num, 1).to(self._xyz.device)
                
                self._xyz = nn.Parameter(torch.cat((self._xyz, new_xyz), dim=0))
                self._cholesky = nn.Parameter(torch.cat((self._cholesky, new_cholesky), dim=0))
                self._features_dc = nn.Parameter(torch.cat((self._features_dc, new_features_dc), dim=0))
                self.rgb_W = torch.nn.Parameter(torch.cat((self.rgb_W, new_rgb_W), dim=0))
     
                for param_group in self.optimizer.param_groups:
                     param_group['params'] = [p for p in self.parameters() if p.requires_grad]
     
                render_pkg1 = self.forward()
    
        # After the first iteration, add new points if needed (from 2nd iteration onward)
        elif 1 < iter < (0.8 * self.init_num_points):
            add_num = int(self.init_num_points * self.add_rate)
            # New points: randomly initialize
            new_xyz = torch.atanh(2 * (torch.rand(add_num, 2) - 0.5)).to(self._xyz.device)
            new_cholesky = torch.rand(add_num, 3).to(self._xyz.device)
            new_features_dc = torch.rand(add_num, 3).to(self._xyz.device)
            new_rgb_W = 0.01 * torch.ones(add_num, 1).to(self._xyz.device)
    
            # Merge original points and new points
            self._xyz = nn.Parameter(torch.cat((self._xyz, new_xyz), dim=0))
            self._cholesky = nn.Parameter(torch.cat((self._cholesky, new_cholesky), dim=0))
            self._features_dc = nn.Parameter(torch.cat((self._features_dc, new_features_dc), dim=0))
            self.rgb_W = torch.nn.Parameter(torch.cat((self.rgb_W, new_rgb_W), dim=0))
    
            # Update optimizer parameters
            for param_group in self.optimizer.param_groups:
                param_group['params'] = [p for p in self.parameters() if p.requires_grad]
    
            render_pkg1 = self.forward()
    
            # Calculate rgb_weight
            rgb_weight = torch.norm(self.rgb_W, dim=1)
            _, sorted_indices = torch.sort(rgb_weight,descending=False)
    
            remove_count = add_num
            with torch.no_grad():
                # Identify which points to remove
                remove_indices = sorted_indices[:remove_count]
                keep_indices = torch.ones(self._xyz.shape[0], dtype=torch.bool, device=self._xyz.device)
                keep_indices[remove_indices] = False
    
                # Retain only the points we want to keep
                self._xyz = nn.Parameter(self._xyz[keep_indices])
                self._cholesky = nn.Parameter(self._cholesky[keep_indices])
                self._features_dc = nn.Parameter(self._features_dc[keep_indices])
                self.rgb_W = torch.nn.Parameter(self.rgb_W[keep_indices])
    
            self.update_optimizer()
    
        # Perform the second forward pass after updating points
        render_pkg2 = self.forward()
        image = render_pkg2["render"]
    
        # Compute the loss and psnr
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.8)
        loss.backward()
    
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
    
        # Update optimizer
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
    
        return loss, psnr
    
    def train_iter(self, gt_image, iter):
        # 第一步：判断是否需要增加高斯点
        if iter < (0.8 * self.init_num_points):
            add_num = int(self.init_num_points * self.add_rate)
            # 新增点：随机初始化
            new_xyz = torch.atanh(2 * (torch.rand(add_num, 2) - 0.5)).to(self._xyz.device)
            new_cholesky = torch.rand(add_num, 3).to(self._xyz.device)
            new_features_dc = torch.rand(add_num, 3).to(self._xyz.device)
            new_rgb_W = 0.01 * torch.ones(add_num, 1).to(self._xyz.device)
            # 合并原始点和新点
            self._xyz = nn.Parameter(torch.cat((self._xyz, new_xyz), dim=0))
            self._cholesky = nn.Parameter(torch.cat((self._cholesky, new_cholesky), dim=0))
            self._features_dc = nn.Parameter(torch.cat((self._features_dc, new_features_dc), dim=0))
            self.rgb_W = torch.nn.Parameter(torch.cat((self.rgb_W, new_rgb_W), dim=0))
            for param_group in self.optimizer.param_groups:
                param_group['params'] = [p for p in self.parameters() if p.requires_grad]  
            render_pkg1 = self.forward()  
            
            rgb_weight = torch.norm(self.rgb_W, dim=1)
            _, sorted_indices = torch.sort(rgb_weight)#排列
            remove_count = add_num
            with torch.no_grad():
                remove_indices = sorted_indices[:add_num]
                keep_indices = torch.ones(self._xyz.shape[0], dtype=torch.bool, device=self._xyz.device)
                keep_indices[remove_indices] = False
                # 保留这些点（裁剪变量）
                self._xyz = nn.Parameter(self._xyz[keep_indices])
                self._cholesky = nn.Parameter(self._cholesky[keep_indices])
                self._features_dc = nn.Parameter(self._features_dc[keep_indices])
                self.rgb_W = torch.nn.Parameter(self.rgb_W[keep_indices]) 
            self.update_optimizer()
     
            # 然后重新 forward，开始训练
        render_pkg2 = self.forward()
        image = render_pkg2["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.8)
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return loss, psnr


    def forward_quantize(self):
        l_vqm, m_bit = 0, 16*self.init_num_points*2 # l_vqm 位置向量量化损失
        # m_bit 是位置向量的比特开销每个点的 xyz 有 3 个坐标，估计是按 16bit 编码（前乘），每个点有 2 个坐标维度参与
        means = torch.tanh(self.xyz_quantizer(self._xyz))#(-1,1)，量化后的坐标
        cholesky_elements, l_vqs, s_bit = self.cholesky_quantizer(self._cholesky)#l_vqs计算损失和s_bit比特开销
        cholesky_elements = cholesky_elements + self.cholesky_bound
        l_vqr, r_bit = 0, 0
        colors, l_vqc, c_bit = self.features_dc_quantizer(self.get_features)#get_features每个点的RGB颜色值
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(means, cholesky_elements, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                colors, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        vq_loss = l_vqm + l_vqs + l_vqr + l_vqc
        return {"render": out_img, "vq_loss": vq_loss, "unit_bit":[m_bit, s_bit, r_bit, c_bit]}

    def train_iter_quantize(self, gt_image):
        render_pkg = self.forward_quantize()
        image = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7) + render_pkg["vq_loss"]
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return loss, psnr

    def compress_wo_ec(self):
        means = torch.tanh(self.xyz_quantizer(self._xyz))
        quant_cholesky_elements, cholesky_elements = self.cholesky_quantizer.compress(self._cholesky)
        cholesky_elements = cholesky_elements + self.cholesky_bound
        colors, feature_dc_index = self.features_dc_quantizer.compress(self.get_features)
        return {"xyz":self._xyz.half(), "feature_dc_index": feature_dc_index, "quant_cholesky_elements": quant_cholesky_elements,}

    def decompress_wo_ec(self, encoding_dict):
        xyz, feature_dc_index, quant_cholesky_elements = encoding_dict["xyz"], encoding_dict["feature_dc_index"], encoding_dict["quant_cholesky_elements"]
        means = torch.tanh(xyz.float())
        cholesky_elements = self.cholesky_quantizer.decompress(quant_cholesky_elements)
        cholesky_elements = cholesky_elements + self.cholesky_bound
        colors = self.features_dc_quantizer.decompress(feature_dc_index)
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(means, cholesky_elements, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                colors, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render":out_img}

    def analysis_wo_ec(self, encoding_dict):
        quant_cholesky_elements, feature_dc_index = encoding_dict["quant_cholesky_elements"], encoding_dict["feature_dc_index"]
        total_bits = 0
        initial_bits, codebook_bits = 0, 0
        for quantizer_index, layer in enumerate(self.features_dc_quantizer.quantizer.layers):
            codebook_bits += layer._codebook.embed.numel()*torch.finfo(layer._codebook.embed.dtype).bits
        initial_bits += self.cholesky_quantizer.scale.numel()*torch.finfo(self.cholesky_quantizer.scale.dtype).bits
        initial_bits += self.cholesky_quantizer.beta.numel()*torch.finfo(self.cholesky_quantizer.beta.dtype).bits
        initial_bits += codebook_bits

        total_bits += initial_bits
        total_bits += self._xyz.numel()*16

        feature_dc_index = feature_dc_index.int().cpu().numpy()
        index_max = np.max(feature_dc_index)
        max_bit = np.ceil(np.log2(index_max)) #calculate max bit for feature_dc_index
        total_bits += feature_dc_index.size * max_bit #get_np_size(encoding_dict["feature_dc_index"]) * 8
        
        quant_cholesky_elements = quant_cholesky_elements.cpu().numpy()
        total_bits += quant_cholesky_elements.size * 6 #cholesky bits 

        position_bits = self._xyz.numel()*16
        cholesky_bits, feature_dc_bits = 0, 0
        cholesky_bits += self.cholesky_quantizer.scale.numel()*torch.finfo(self.cholesky_quantizer.scale.dtype).bits
        cholesky_bits += self.cholesky_quantizer.beta.numel()*torch.finfo(self.cholesky_quantizer.beta.dtype).bits
        cholesky_bits += quant_cholesky_elements.size * 6
        feature_dc_bits += codebook_bits
        feature_dc_bits += feature_dc_index.size * max_bit

        bpp = total_bits/self.H/self.W/3
        position_bpp = position_bits/self.H/self.W/3
        cholesky_bpp = cholesky_bits/self.H/self.W/3
        feature_dc_bpp = feature_dc_bits/self.H/self.W/3
        return {"bpp": bpp, "position_bpp": position_bpp, 
            "cholesky_bpp": cholesky_bpp, "feature_dc_bpp": feature_dc_bpp}

    def compress(self):
        means = torch.tanh(self.xyz_quantizer(self._xyz))
        quant_cholesky_elements, cholesky_elements = self.cholesky_quantizer.compress(self._cholesky)
        cholesky_elements = cholesky_elements + self.cholesky_bound
        colors, feature_dc_index = self.features_dc_quantizer.compress(self.get_features)
        cholesky_compressed, cholesky_histogram_table, cholesky_unique = compress_matrix_flatten_categorical(quant_cholesky_elements.int().flatten().tolist())
        feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique = compress_matrix_flatten_categorical(feature_dc_index.int().flatten().tolist())
        return {"xyz":self._xyz.half(), "feature_dc_index": feature_dc_index, "quant_cholesky_elements": quant_cholesky_elements, 
            "feature_dc_bitstream":[feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique], 
            "cholesky_bitstream":[cholesky_compressed, cholesky_histogram_table, cholesky_unique]}

    def decompress(self, encoding_dict):
        xyz = encoding_dict["xyz"]
        num_points, device = xyz.size(0), xyz.device
        feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique = encoding_dict["feature_dc_bitstream"]
        cholesky_compressed, cholesky_histogram_table, cholesky_unique = encoding_dict["cholesky_bitstream"]
        feature_dc_index = decompress_matrix_flatten_categorical(feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique, num_points*2, (num_points, 2))
        quant_cholesky_elements = decompress_matrix_flatten_categorical(cholesky_compressed, cholesky_histogram_table, cholesky_unique, num_points*3, (num_points, 3))
        feature_dc_index = torch.from_numpy(feature_dc_index).to(device).int() #[800, 2]
        quant_cholesky_elements = torch.from_numpy(quant_cholesky_elements).to(device).float() #[800, 3]

        means = torch.tanh(xyz.float())
        cholesky_elements = self.cholesky_quantizer.decompress(quant_cholesky_elements)
        cholesky_elements = cholesky_elements + self.cholesky_bound
        colors = self.features_dc_quantizer.decompress(feature_dc_index)
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(means, cholesky_elements, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                colors, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render":out_img}
   
    def analysis(self, encoding_dict):
        quant_cholesky_elements, feature_dc_index = encoding_dict["quant_cholesky_elements"], encoding_dict["feature_dc_index"]
        cholesky_compressed, cholesky_histogram_table, cholesky_unique = compress_matrix_flatten_categorical(quant_cholesky_elements.int().flatten().tolist())
        feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique = compress_matrix_flatten_categorical(feature_dc_index.int().flatten().tolist())  
        cholesky_lookup = dict(zip(cholesky_unique, cholesky_histogram_table.astype(np.float64) / np.sum(cholesky_histogram_table).astype(np.float64)))
        feature_dc_lookup = dict(zip(feature_dc_unique, feature_dc_histogram_table.astype(np.float64) / np.sum(feature_dc_histogram_table).astype(np.float64)))

        total_bits = 0
        initial_bits, codebook_bits = 0, 0
        for quantizer_index, layer in enumerate(self.features_dc_quantizer.quantizer.layers):
            codebook_bits += layer._codebook.embed.numel()*torch.finfo(layer._codebook.embed.dtype).bits
        initial_bits += self.cholesky_quantizer.scale.numel()*torch.finfo(self.cholesky_quantizer.scale.dtype).bits
        initial_bits += self.cholesky_quantizer.beta.numel()*torch.finfo(self.cholesky_quantizer.beta.dtype).bits
        initial_bits += get_np_size(cholesky_histogram_table) * 8
        initial_bits += get_np_size(cholesky_unique) * 8 
        initial_bits += get_np_size(feature_dc_histogram_table) * 8
        initial_bits += get_np_size(feature_dc_unique) * 8  
        initial_bits += codebook_bits

        total_bits += initial_bits
        total_bits += self._xyz.numel()*16
        total_bits += get_np_size(cholesky_compressed) * 8
        total_bits += get_np_size(feature_dc_compressed) * 8

        position_bits = self._xyz.numel()*16
        cholesky_bits, feature_dc_bits = 0, 0
        cholesky_bits += self.cholesky_quantizer.scale.numel()*torch.finfo(self.cholesky_quantizer.scale.dtype).bits
        cholesky_bits += self.cholesky_quantizer.beta.numel()*torch.finfo(self.cholesky_quantizer.beta.dtype).bits
        cholesky_bits += get_np_size(cholesky_histogram_table) * 8
        cholesky_bits += get_np_size(cholesky_unique) * 8   
        cholesky_bits += get_np_size(cholesky_compressed) * 8
        feature_dc_bits += codebook_bits
        feature_dc_bits += get_np_size(feature_dc_histogram_table) * 8
        feature_dc_bits += get_np_size(feature_dc_unique) * 8  
        feature_dc_bits += get_np_size(feature_dc_compressed) * 8

        bpp = total_bits/self.H/self.W/3
        position_bpp = position_bits/self.H/self.W/3
        cholesky_bpp = cholesky_bits/self.H/self.W/3
        feature_dc_bpp = feature_dc_bits/self.H/self.W/3
        return {"bpp": bpp, "position_bpp": position_bpp, 
            "cholesky_bpp": cholesky_bpp, "feature_dc_bpp": feature_dc_bpp,}
