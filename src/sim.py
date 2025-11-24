import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim


class SIMoptimizer:
    """
    Stacked Intelligent Metasurface (SIM) Optimizer

    This class simulates a SIM and optimizes its internal phase shifts to approximate
    a given target matrix A in the wave domain using electromagnetic wave propagation models.

    The optimization minimizes the Frobenius norm of the fitting error between the SIM's
    overall forward propagation matrix G (scaled by β) and a target matrix A.

    Supports different input and output plane dimensions.
    """

    def __init__(
        self,
        num_intermediate_layers: int,
        num_meta_atoms_input_x: int,
        num_meta_atoms_input_y: int,
        num_meta_atoms_output_x: int,
        num_meta_atoms_output_y: int,
        num_meta_atoms_intermediate_x: int,
        num_meta_atoms_intermediate_y: int,
        sim_thickness: float,
        wavelength: float,
        meta_atom_spacing_intermediate_x: float,
        meta_atom_spacing_intermediate_y: float,
        meta_atom_spacing_input_x: float,
        meta_atom_spacing_input_y: float,
        meta_atom_spacing_output_x: float,
        meta_atom_spacing_output_y: float,
        verbose: bool = False,
    ):
        """
        Initialize the SIM optimizer with different input/output dimensions.

        Args:
            num_intermediate_layers: Number of intermediate metasurface layers (L)
            num_meta_atoms_input_x: Meta-atoms in x-direction on input plane
            num_meta_atoms_input_y: Meta-atoms in y-direction on input plane
            num_meta_atoms_output_x: Meta-atoms in x-direction on output plane
            num_meta_atoms_output_y: Meta-atoms in y-direction on output plane
            num_meta_atoms_intermediate_x: Meta-atoms in x-direction on intermediate layers
            num_meta_atoms_intermediate_y: Meta-atoms in y-direction on intermediate layers
            sim_thickness: Total physical thickness of SIM in meters (TSIM)
            wavelength: Operating wavelength in meters (λ)
            meta_atom_spacing_intermediate_x: Spacing between meta-atoms on intermediate layers (sx)
            meta_atom_spacing_intermediate_y: Spacing between meta-atoms on intermediate layers (sy)
            meta_atom_spacing_input_x: Spacing between meta-atoms on input plane (dx_in)
            meta_atom_spacing_input_y: Spacing between meta-atoms on input plane (dy_in)
            meta_atom_spacing_output_x: Spacing between meta-atoms on output plane (dx_out)
            meta_atom_spacing_output_y: Spacing between meta-atoms on output plane (dy_out)
            verbose: allowing prints during training.
        """
        # Validate inputs
        if num_intermediate_layers <= 0:
            raise ValueError('Number of intermediate layers must be positive')
        if sim_thickness <= 0 or wavelength <= 0:
            raise ValueError('Thickness and wavelength must be positive')

        # Store parameters
        self.L = num_intermediate_layers
        self.Nx_in = num_meta_atoms_input_x
        self.Ny_in = num_meta_atoms_input_y
        self.Nx_out = num_meta_atoms_output_x
        self.Ny_out = num_meta_atoms_output_y
        self.Mx_int = num_meta_atoms_intermediate_x
        self.My_int = num_meta_atoms_intermediate_y
        self.N_in = self.Nx_in * self.Ny_in
        self.N_out = self.Nx_out * self.Ny_out
        self.M_int = self.Mx_int * self.My_int
        self.sim_thickness = sim_thickness
        self.wavelength = wavelength
        self.sx = meta_atom_spacing_intermediate_x
        self.sy = meta_atom_spacing_intermediate_y
        self.dx_in = meta_atom_spacing_input_x
        self.dy_in = meta_atom_spacing_input_y
        self.dx_out = meta_atom_spacing_output_x
        self.dy_out = meta_atom_spacing_output_y
        self.verbose = verbose

        # Derived parameters
        self.slayer = (
            sim_thickness / num_intermediate_layers
        )  # Vertical spacing between layers
        self.kappa = 2 * np.pi / wavelength  # Wavenumber

        # Initialize random phase shifts for intermediate layers [0, 2π)
        self._phase_shifts = [
            np.random.uniform(0, 2 * np.pi, self.M_int) for _ in range(self.L)
        ]

        # Pre-calculate fixed attenuation matrices
        self.W0 = self._calculate_attenuation_matrix_for_layer_pair(
            'input', 'intermediate'
        )
        self.W_L = self._calculate_attenuation_matrix_for_layer_pair(
            'intermediate', 'output'
        )

        # Pre-calculate intermediate layer attenuation matrices W_1 to W_{L-1}
        self.W_intermediate = []
        for _ in range(1, self.L):  # W_1 to W_{L-1}
            W_l = self._calculate_attenuation_matrix_for_layer_pair(
                'intermediate', 'intermediate'
            )
            self.W_intermediate.append(W_l)

    def _get_2d_coords(
        self, linear_idx: int, num_atoms_x: int
    ) -> tuple[int, int]:
        """
        Convert 1-indexed linear meta-atom index to 1-indexed (x, y) coordinates.

        Args:
            linear_idx: 1-indexed linear index
            num_atoms_x: Number of atoms in x-direction

        Returns:
            tuple of (nx, ny) as 1-indexed coordinates
        """
        my = int(np.ceil(linear_idx / num_atoms_x))
        mx = linear_idx - (my - 1) * num_atoms_x
        return mx, my

    def _calculate_propagation_distance(
        self,
        from_layer_num_atoms_x: int,
        from_layer_num_atoms_y: int,
        from_layer_spacing_x: float,
        from_layer_spacing_y: float,
        from_linear_idx: int,
        to_layer_num_atoms_x: int,
        to_layer_num_atoms_y: int,
        to_layer_spacing_x: float,
        to_layer_spacing_y: float,
        to_linear_idx: int,
    ) -> float:
        """
        Calculate propagation distance between two meta-atoms on adjacent layers.

        Returns:
            Distance d_m,m̆ between the meta-atoms
        """
        # Get 2D coordinates for both atoms
        from_x, from_y = self._get_2d_coords(
            from_linear_idx, from_layer_num_atoms_x
        )
        to_x, to_y = self._get_2d_coords(to_linear_idx, to_layer_num_atoms_x)

        # Calculate horizontal distance differences using 'from' layer spacing
        dx_diff = (from_x - to_x) * from_layer_spacing_x
        dy_diff = (from_y - to_y) * from_layer_spacing_y

        # Calculate 3D distance including vertical separation
        distance = np.sqrt(dx_diff**2 + dy_diff**2 + self.slayer**2)
        return distance

    def _calculate_attenuation_matrix_for_layer_pair(
        self, from_layer_type: str, to_layer_type: str
    ) -> np.ndarray:
        """
        Calculate attenuation matrix for propagation between two layer types.

        Args:
            from_layer_type: "input", "intermediate", or "output"
            to_layer_type: "input", "intermediate", or "output"

        Returns:
            Attenuation matrix with appropriate dimensions
        """
        # Determine layer parameters
        layer_params = {
            'input': (
                self.N_in,
                self.Nx_in,
                self.Ny_in,
                self.dx_in,
                self.dy_in,
            ),
            'intermediate': (
                self.M_int,
                self.Mx_int,
                self.My_int,
                self.sx,
                self.sy,
            ),
            'output': (
                self.N_out,
                self.Nx_out,
                self.Ny_out,
                self.dx_out,
                self.dy_out,
            ),
        }

        from_total, from_nx, from_ny, from_sx, from_sy = layer_params[
            from_layer_type
        ]
        to_total, to_nx, to_ny, to_sx, to_sy = layer_params[to_layer_type]

        # Meta-atom area of the sending layer
        A_meta_atom = from_sx * from_sy

        # Initialize attenuation matrix
        W = np.zeros((to_total, from_total), dtype=complex)

        # Calculate attenuation for each pair of meta-atoms
        for to_idx in range(1, to_total + 1):  # 1-indexed
            for from_idx in range(1, from_total + 1):  # 1-indexed
                # Calculate propagation distance
                d = self._calculate_propagation_distance(
                    from_nx,
                    from_ny,
                    from_sx,
                    from_sy,
                    from_idx,
                    to_nx,
                    to_ny,
                    to_sx,
                    to_sy,
                    to_idx,
                )

                # Apply attenuation formula
                if d > 0:  # Avoid division by zero
                    attenuation = (
                        A_meta_atom
                        * self.slayer
                        / (2 * np.pi * d**3)
                        * (1 - 1j * self.kappa * d)
                        * np.exp(1j * self.kappa * d)
                    )
                    W[to_idx - 1, from_idx - 1] = attenuation

        return W

    def _get_intermediate_Y_matrices(self) -> list[np.ndarray]:
        """
        Convert stored phase shifts to diagonal Υ matrices.

        Returns:
        """
        Y_matrices = []
        for layer in range(self.L):
            Y_l = np.diag(np.exp(1j * self._phase_shifts[layer]))
            Y_matrices.append(Y_l)
        return Y_matrices

    def _calculate_sim_propagation_G(self) -> np.ndarray:
        """
        Calculate overall forward propagation matrix G.

        Returns:
            G matrix representing SIM forward propagation (N_out x N_in)
        """
        Y_matrices = self._get_intermediate_Y_matrices()

        # Start with the rightmost operation: W_0
        G = self.W0.copy()

        # Apply Υ_1 * W_0
        G = Y_matrices[0] @ G

        # Apply intermediate layers: W_l * Υ_{l+1} for l = 1 to L-1
        for layer in range(1, self.L):
            if layer - 1 < len(self.W_intermediate):
                G = self.W_intermediate[layer - 1] @ G
            G = Y_matrices[layer] @ G

        # Apply final W_L
        G = self.W_L @ G

        return G

    def _generate_target_2d_matrix(self) -> np.ndarray:
        """
        Generate standard 2D matrix A for square input/output case.

        Returns:
            2D matrix (only works when N_in = N_out and both are square)
        """
        if self.N_in != self.N_out:
            raise ValueError(
                'Standard 2D only works for equal input/output dimensions'
            )
        if self.Nx_in != self.Ny_in:
            raise ValueError('Standard 2D requires square input array')

        A = np.zeros((self.N_out, self.N_in), dtype=complex)

        for n in range(1, self.N_out + 1):  # 1-indexed
            for n_hat in range(1, self.N_in + 1):  # 1-indexed
                # Get 2D coordinates
                nx, ny = self._get_2d_coords(n, self.Nx_out)
                n_hat_x, n_hat_y = self._get_2d_coords(n_hat, self.Nx_in)

                # Calculate matrix entry
                f_n_n_hat = np.exp(
                    -1j * 2 * np.pi * (nx - 1) * (n_hat_x - 1) / self.Nx_in
                ) * np.exp(
                    -1j * 2 * np.pi * (ny - 1) * (n_hat_y - 1) / self.Ny_in
                )

                A[n - 1, n_hat - 1] = f_n_n_hat

        return A

    def _calculate_gradient_for_layer(
        self,
        G_current: np.ndarray,
        A_target: np.ndarray,
        beta_current: complex,
        layer_idx_1_based: int,
    ) -> np.ndarray:
        """
        Calculate gradient vector ∇ξl L for a specific intermediate layer.

        Args:
            G_current: Current G matrix (N_out x N_in)
            A_target: Target A matrix (N_out x N_in)
            beta_current: Current scaling factor β
            layer_idx_1_based: Layer index (1-based, from 1 to L)

        Returns:
            Gradient vector for the specified layer
        """
        l_idx = layer_idx_1_based  # 1-based layer index
        Y_matrices = self._get_intermediate_Y_matrices()
        # gradient = np.zeros(self.M_int, dtype=complex)
        gradient = np.zeros(self.M_int)

        for n in range(1, self.N_in + 1):  # 1-indexed column (input dimension)
            # Get w0_n: n-th column of W0
            w0_n = self.W0[:, n - 1]

            # Calculate q_l_n = W_{l-1} * Υ_{l-1} * ... * W_1 * Υ_1 * w0_n
            q_l_n = w0_n.copy()

            # Apply transformations up to layer l-1
            for layer in range(1, l_idx):  # 1 to l-1
                q_l_n = Y_matrices[layer - 1] @ q_l_n  # Apply Υ_layer
                if layer < self.L and layer - 1 < len(self.W_intermediate):
                    q_l_n = (
                        self.W_intermediate[layer - 1] @ q_l_n
                    )  # Apply W_layer

            # Calculate P_l_n = W_L * Υ_L * ... * W_{l+1} * Υ_{l+1} * W_l * diag(q_l_n)
            P_l_n = np.diag(q_l_n)

            # Apply transformations from layer l+1 to L
            for layer in range(l_idx + 1, self.L + 1):  # l+1 to L
                if layer <= self.L and layer - 2 < len(self.W_intermediate):
                    P_l_n = (
                        self.W_intermediate[layer - 2] @ P_l_n
                    )  # Apply W_{layer-1}
                if layer <= self.L:
                    P_l_n = Y_matrices[layer - 1] @ P_l_n  # Apply Υ_layer

            # Apply final W_L
            P_l_n = self.W_L @ P_l_n

            # Get column vectors
            g_n = G_current[:, n - 1]
            a_n = A_target[:, n - 1]

            # Calculate intermediate term for gradient sum
            # term = np.sum(P_l_n * (beta_current * g_n - a_n)[:, np.newaxis], axis=0)
            term = np.imag(
                beta_current.conj()
                * Y_matrices[l_idx - 1].conj().T
                @ P_l_n.conj().T
                @ (beta_current * g_n - a_n)[:, np.newaxis]
            )
            gradient += term.flatten()

        # Apply final factor: 2 * Imag(beta_current.conj() * ...)
        # gradient = 2 * np.imag(beta_current.conj() * gradient)
        gradient *= 2

        return gradient

    def calculate_optimal_beta(
        self, G_current: np.ndarray, A_target: np.ndarray
    ) -> complex:
        """
        Calculate optimal scaling factor β using least squares method.

        The optimal β minimizes ||βG - A||²_F and is given by:
        β = (g^H g)^{-1} g^H a
        where g = vec(G) and a = vec(A), and g^H is the Hermitian (conjugate transpose) of g

        Args:
            G_current: Current SIM propagation matrix (N_out x N_in)
            A_target: Target matrix (N_out x N_in)

        Returns:
            Optimal complex scaling factor β
        """
        # Vectorize matrices
        g = G_current.reshape(-1, 1, order='F')  # vec(G) - column vector
        a = A_target.reshape(-1, 1, order='F')  # vec(A) - column vector

        # Calculate least squares solution: β = (g^H g)^{-1} g^H a
        beta_optimal = np.linalg.inv(g.conj().T @ g) @ (g.conj().T @ a)

        return beta_optimal.flatten()

    def optimize_sim_for_target_matrix(
        self,
        A_target: np.ndarray,
        max_iterations: int,
        learning_rate_initial: float,
        learning_rate_decay: float,
        return_loss_curve: bool = False,  # New flag
    ) -> tuple[list[np.ndarray], float, list[float]]:
        """
        Optimize SIM phase shifts to approximate a given target matrix A.

        Args:
            A_target: Target matrix to approximate (N_out x N_in)
            max_iterations: Maximum number of optimization iterations
            learning_rate_initial: Initial learning rate
            learning_rate_decay: Learning rate decay factor per iteration
            return_loss_curve: Whether to return the full list of training losses

        Returns:
            tuple of (best_phase_shifts, best_loss, [loss history] if return_loss_curve is True)
        """
        if A_target.shape != (self.N_out, self.N_in):
            raise ValueError(
                f'A_target must have shape ({self.N_out}, {self.N_in}), '
                f'got {A_target.shape}'
            )

        current_learning_rate = learning_rate_initial
        best_loss = float('inf')
        best_phase_shifts = None
        loss_history = []

        print(f'Starting optimization with {max_iterations} iterations...')

        for iteration in tqdm(range(max_iterations)):
            G_current = self._calculate_sim_propagation_G()
            row, col = G_current.shape
            beta = self.calculate_optimal_beta(G_current, A_target)
            current_loss = (
                np.linalg.norm(beta * G_current - A_target, 'fro') ** 2
            )
            current_loss /= row * col
            loss_history.append(current_loss)

            if current_loss < best_loss:
                best_loss = current_loss
                best_phase_shifts = [np.copy(ps) for ps in self._phase_shifts]

            if iteration % 10 == 0:
                print(
                    f'Iteration {iteration:3d}: Loss = {current_loss:.6f}, '
                    f'Best = {best_loss:.6f}, |β| = {abs(beta.item()):.4f}'
                )

            maximum = 0
            for l_idx in range(self.L):
                gradient_l = self._calculate_gradient_for_layer(
                    G_current, A_target, beta, l_idx + 1
                )
                self._phase_shifts[l_idx] -= current_learning_rate * gradient_l
                self._phase_shifts[l_idx] = np.fmod(
                    self._phase_shifts[l_idx], 2 * np.pi
                )
                self._phase_shifts[l_idx] = np.where(
                    self._phase_shifts[l_idx] < 0,
                    self._phase_shifts[l_idx] + 2 * np.pi,
                    self._phase_shifts[l_idx],
                )
                if current_max := np.max(np.abs(gradient_l)) > maximum:
                    maximum = current_max

            current_learning_rate *= learning_rate_decay * (np.pi / maximum)

        print(f'Optimization completed. Best loss: {best_loss:.6f}')

        if return_loss_curve:
            return best_phase_shifts, best_loss, loss_history
        else:
            return best_phase_shifts, best_loss

    # Legacy method for backward compatibility
    def optimize_sim(
        self,
        max_iterations: int,
        learning_rate_initial: float,
        learning_rate_decay: float,
    ) -> tuple[list[np.ndarray], float]:
        """
        Optimize SIM phase shifts to approximate 2D matrix (legacy method).
        Only works when input and output dimensions are equal and square.
        """
        A_target = self._generate_target_2d_matrix()
        return self.optimize_sim_for_target_matrix(
            A_target,
            max_iterations,
            learning_rate_initial,
            learning_rate_decay,
        )


class SIMoptimizerTorch(SIMoptimizer, nn.Module):
    def __init__(self, *args, **kwargs):
        SIMoptimizer.__init__(
            self, *args, **kwargs
        )  # call parent init explicitly
        nn.Module.__init__(self)  # and nn.Module init

        # Convert precomputed matrices to torch tensors (complex dtype)
        self.W0 = torch.tensor(self.W0, dtype=torch.cfloat)
        self.W_L = torch.tensor(self.W_L, dtype=torch.cfloat)
        self.W_intermediate = [
            torch.tensor(W, dtype=torch.cfloat) for W in self.W_intermediate
        ]

        # Make phases learnable
        self.phase_shifts = nn.Parameter(torch.rand(self.L, self.M_int))

    def _get_intermediate_Y_matrices(self):
        """Build diagonal Υ matrices from phase parameters."""
        phase_shifts = 2 * torch.pi * torch.sigmoid(self.phase_shifts)

        Y_matrices = []
        for layer in range(self.L):
            phase = phase_shifts[layer]
            Y_l = torch.diag(torch.exp(1j * phase))
            Y_matrices.append(Y_l)
        return Y_matrices

    def _calculate_sim_propagation_G(self):
        """Compute SIM propagation matrix G using torch operations."""
        Y_matrices = self._get_intermediate_Y_matrices()
        G = self.W0.clone()

        G = Y_matrices[0] @ G

        for l_idx in range(1, self.L):
            if l_idx - 1 < len(self.W_intermediate):
                G = self.W_intermediate[l_idx - 1] @ G
            G = Y_matrices[l_idx] @ G

        G = self.W_L @ G
        return G

    def optimize_with_torch(self, A_target, max_iterations=200, lr=1e-2):
        """
        Optimize SIM phase shifts using PyTorch autograd.
        """
        A_target = torch.tensor(A_target, dtype=torch.cdouble)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_history = []

        for it in tqdm(range(max_iterations)):
            optimizer.zero_grad()

            G = self._calculate_sim_propagation_G()
            beta = self.calculate_optimal_beta(
                G.detach().cpu().numpy(), A_target.cpu().resolve_conj().numpy()
            )
            beta = torch.from_numpy(beta).to(torch.complex64)
            loss = torch.norm(beta * G - A_target, p='fro') ** 2 / (
                self.N_out * self.N_in
            )

            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            if it % 10 == 0 and self.verbose:
                print(
                    f'Iter {it:3d}: Loss={loss.item():.6e}, |β|={abs(beta.item()):.3f}'
                )

        phase_shifts = 2 * torch.pi * torch.sigmoid(self.phase_shifts)
        phase_shifts = [p.detach().numpy() for p in phase_shifts]
        return phase_shifts, loss_history


def main() -> None:
    """The main loop."""
    return None


if __name__ == '__main__':
    main()
