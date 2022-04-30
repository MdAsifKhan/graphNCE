import torch.nn as nn
import torch
from torch_geometric.transforms import GDC
from torch_geometric.utils import add_self_loops, is_undirected, to_dense_adj, get_laplacian

class DiffusionAugmentation(nn.Module):
    def __init__(self, nm_filters = 8,
                        filter_type = 'wavelet',
                        add_self_loop = True):
        super(DiffusionAugmentation, self).__init__()
        self.nm_filters = nm_filters
        self.filter_type = filter_type
        #self.mixingdistn = Dirichlet(torch.tensor([0.5]*nm_scale))

    def transition_matrix(self, edge_index, edge_weight, num_nodes,
                          normalization):
        r"""Calculate the approximate, sparse diffusion on a given sparse
        matrix.

        Args:
            edge_index (LongTensor): The edge indices.
            edge_weight (Tensor): One-dimensional edge weights.
            num_nodes (int): Number of nodes.
            normalization (str): Normalization scheme:

                1. :obj:`"sym"`: Symmetric normalization
                   :math:`\mathbf{T} = \mathbf{D}^{-1/2} \mathbf{A}
                   \mathbf{D}^{-1/2}`.
                2. :obj:`"col"`: Column-wise normalization
                   :math:`\mathbf{T} = \mathbf{A} \mathbf{D}^{-1}`.
                3. :obj:`"row"`: Row-wise normalization
                   :math:`\mathbf{T} = \mathbf{D}^{-1} \mathbf{A}`.
                4. :obj:`None`: No normalization.

        :rtype: (:class:`LongTensor`, :class:`Tensor`)
        """
        if normalization == 'sym':
            row, col = edge_index
            deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        elif normalization == 'col':
            _, col = edge_index
            deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
            deg_inv = 1. / deg
            deg_inv[deg_inv == float('inf')] = 0
            edge_weight = edge_weight * deg_inv[col]
        elif normalization == 'row':
            row, _ = edge_index
            deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
            deg_inv = 1. / deg
            deg_inv[deg_inv == float('inf')] = 0
            edge_weight = edge_weight * deg_inv[row]
        elif normalization is None:
            pass
        else:
            raise ValueError(
                f"Transition matrix normalization '{normalization}' unknown")

        return edge_index, edge_weight

    def heat_diffusion(self, edge_index, edge_weight, eps=0.1):
        nm_nodes = edge_index.max().item() + 1
        T = torch.logspace(0.01, 1, self.nm_filters)
        edge_index_T, edge_weight_T = [], []
        for t in T:
            diffusion_t = GDC().diffusion_matrix_exact(edge_index, edge_weight, method='heat', t=t)
            edge_index_t, edge_weight_t = GDC().sparsify_dense(diffusion_t, method='threshold', eps=eps)
            edge_index_t, edge_weight_t = coalesce(edge_index_t, edge_weight_t, nm_nodes, nm_nodes)
            edge_index_t, edge_weight_t = self.transition_matrix(edge_index_t, edge_weight_t, nm_nodes, normalization='sym')
            edge_index_T.append(edge_index_t)
            edge_weight_T.append(edge_weight_t)
        edge_index_T = torch.stack(edge_index_T)
        edge_weight_T = torch.stack(edge_weight_T)
        return edge_index_T, edge_weight_T

    def wavelet_diffusion(self, edge_index, edge_weight, eps=0.1):
        nm_nodes = edge_index.max().item() + 1
        edge_index, edge_weight = self.transition_matrix(edge_index, edge_weight, nm_nodes, normalization='sym')
        W = to_dense_adj(edge_index, edge_attr=edge_weight).squeeze()
        eye = torch.eye(adj.shape).to(adj.device)
        T = 0.5 * (eye + W) # Lazy diffusion operator
        edge_index_0, edge_weight_0 = GDC().sparsify_dense(T, method='threshold', eps=eps)
        edge_index_0, edge_weight_0 = coalesce(edge_index_0, edge_weight_0, nm_nodes, nm_nodes)
        edge_index_0, edge_weight_0 = self.transition_matrix(edge_index_0, edge_weight_0, nm_nodes, normalization='sym')
        edge_index_T = [edge_index_t]
        edge_weight_T = [edge_weight_t]
0       for i in range(self.nm_scale):
            dilation = 2**(i-1)
            Tp = torch.matrix_power(T, dilation)
            filter_t = Lp.mm(eye - Tp)
            edge_index_t, edge_weight_t = GDC().sparsify_dense(filter_t, method='threshold', eps=eps)
            edge_index_t, edge_weight_t = coalesce(edge_index_t, edge_weight_t, nm_nodes, nm_nodes)
            edge_index, edge_weight = self.transition_matrix(edge_index_t, edge_weight_t, nm_nodes, normalization='sym')
            edge_index_T.append(edge_index_t)
            edge_weight_T.append(edge_weight_t)
        edge_index_T = torch.stack(edge_index_T)
        edge_weight_T = torch.stack(edge_weight_T)
        return edge_index_T, edge_weight_T

    def forward(self, edge_index, edge_weight):
        if self.filter_type == 'heat':
            return self.heat_diffusion(edge_index, edge_weight)
        elif self.filter_type == 'wavelet':
            return self.wavelet_diffusion(edge_index, edge_weight)