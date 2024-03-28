import torch

def solve_for_jacobian(q, p):
    F, _, _ = p.shape
    # Add a column of ones to p for each frame to make it square and avoid singularity
    ones = torch.ones(F, 3, 1, dtype=p.dtype, device=p.device)
    p_augmented = torch.cat((p, ones), dim=2)  # Fx3x3
    
    # Use torch.linalg.solve for the batch, assuming p_augmented is not singular
    J_augmented = torch.linalg.solve(p_augmented, q.unsqueeze(-1))  # Fx3x1
    
    # Take just the first two entries of J for each frame, ignoring the affine term
    J_final = J_augmented[:, :2, :].squeeze(-1)  # Remove the last dimension to get Fx2
    
    return J_final

def energy(uv, q, f):
    p = uv[f]
    j = solve_for_jacobian(q, p)
    j_reshaped = j.view(-1, 1, 2)
    m = torch.bmm(j_reshaped.transpose(1,2), j_reshaped)
    area = compute_area(p)
    m_surface = torch.einsum('f,fik->ik',area, m)
    energy = torch.diagonal(m_surface).sum()
    return energy

def compute_area(p):
    a = p[:,0]
    b = p[:,1]
    c = p[:,2]

    ab = b-a
    ac = c-a

    area = 0.5 * torch.abs(ab[:,0]*ac[:,1] - ab[:,1]*ac[:,0])

    #area /= torch.max(area) #normalized
    return area






