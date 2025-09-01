from matplotlib import image
import nvdiffrast.torch as dr
import torch

def _warmup(glctx, device):
    #windows workaround for https://github.com/NVlabs/nvdiffrast/issues/59
 
    pos = torch.tensor([[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [-0.8, 0.8, 0, 1]]], dtype=torch.float32, device=device)
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)
    dr.rasterize(glctx, pos, tri, resolution=[256, 256])

class NormalsRenderer:
    
    _glctx:dr.RasterizeGLContext = None
    
    def __init__(
            self,
            mv: torch.Tensor, #C,4,4
            proj: torch.Tensor, #C,4,4
            image_size: tuple[int,int],
            device: str
            ):
        self._mvp = proj @ mv #C,4,4
        self._image_size = image_size
        # self._glctx = dr.RasterizeGLContext()
        self._glctx = dr.RasterizeCudaContext(device=device)
        _warmup(self._glctx, device)

    def render(self,
            vertices: torch.Tensor, #V,3 float
            faces: torch.Tensor, #F,3 long
            colors: torch.Tensor = None, #V,3 float
            normals: torch.Tensor = None, #V,3 float
            return_triangles: bool = False  
            ) -> torch.Tensor: #C,H,W,4

        V = vertices.shape[0]
        faces = faces.type(torch.int32)
        vert_hom = torch.cat((vertices, torch.ones(V,1,device=vertices.device)),axis=-1) #V,3 -> V,4
        vertices_clip = vert_hom @ self._mvp.transpose(-2,-1) #C,V,4
        rast_out,_ = dr.rasterize(self._glctx, vertices_clip, faces, resolution=self._image_size, grad_db=False) #C,H,W,4
        vert_nrm = (normals+1)/2 if normals is not None else colors
        nrm, _ = dr.interpolate(vert_nrm, rast_out, faces) #C,H,W,3
        alpha = torch.clamp(rast_out[..., -1:], max=1) #C,H,W,1
        nrm = torch.concat((nrm,alpha),dim=-1) #C,H,W,4
        nrm = dr.antialias(nrm, rast_out, vertices_clip, faces) #C,H,W,4
        if return_triangles:
            return nrm, rast_out[..., -1]
        return nrm #C,H,W,4
            

class LocalNormaRender:
    
    _glctx:dr.RasterizeGLContext = None
    
    def __init__(
            self,
            mv: torch.Tensor, #C,4,4
            proj: torch.Tensor, #C,4,4
            image_size: tuple[int,int],
            device: str
            ):
        self._mv = mv
        self._mvp = proj @ mv #C,4,4
        self._image_size = image_size
        # self._glctx = dr.RasterizeGLContext()
        self._glctx = dr.RasterizeCudaContext(device=device)
        _warmup(self._glctx, device)

    def render(self,
            vertices: torch.Tensor, #V,3 float
            faces: torch.Tensor, #F,3 long
            colors: torch.Tensor = None, #V,3 float
            normals: torch.Tensor = None, #V,3 float
            return_triangles: bool = False  
            ) -> torch.Tensor: #C,H,W,4

        V = vertices.shape[0]
        faces = faces.type(torch.int32)
        vert_hom = torch.cat((vertices, torch.ones(V,1,device=vertices.device)),axis=-1) #V,3 -> V,4
        vertices_clip = vert_hom @ self._mvp.transpose(-2,-1) #C,V,4
        rast_out,_ = dr.rasterize(self._glctx, vertices_clip, faces, resolution=self._image_size, grad_db=False) #C,H,W,4

        if normals is not None:
            # 计算每个视角的法线矩阵（view 的左上 3x3 的逆转置）并应用到顶点法线
            view_R = self._mv[:, :3, :3]                                        # C,3,3
            normal_matrix = torch.inverse(view_R).transpose(-2, -1)             # C,3,3
            nrm_cam = (normals.unsqueeze(0)) @ normal_matrix.transpose(-2, -1)  # C,V,3
            # x轴翻转后优化结果内凹的问题解决（配合stable normal预测的normal图）,也能适用在 gt normal 图上
            nrm_cam[..., :1] = -nrm_cam[..., :1]
            nrm_cam[..., 1:2] = -nrm_cam[..., 1:2]
            nrm_cam = torch.nn.functional.normalize(nrm_cam, dim=-1)
            vert_nrm = (nrm_cam + 1.0) / 2.0                                    # C,V,3 in [0,1]
        else:
            vert_nrm = colors if colors is not None else None

        if vert_nrm is None:
            raise ValueError("Either normals or colors must be provided.")

        nrm, _ = dr.interpolate(vert_nrm, rast_out, faces) #C,H,W,3
        alpha = torch.clamp(rast_out[..., -1:], max=1) #C,H,W,1
        nrm = torch.concat((nrm,alpha),dim=-1) #C,H,W,4
        nrm = dr.antialias(nrm, rast_out, vertices_clip, faces) #C,H,W,4
        if return_triangles:
            return nrm, rast_out[..., -1]
        return nrm #C,H,W,4
