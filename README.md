# gcmc

Code to run GCMC simulations

## About the code

This is a python program for GCMC simulations of fluid with short-ranged potentials. For electrostatic interaction, a Gaussian truncated potential is used. See detailed documentation at the [Wiki](https://github.com/annatbui/gcmc/wiki/Home).

<p align="center">
<img src="https://github.com/user-attachments/assets/7bcb5613-292e-42a3-8be3-eaf49ac52ae3" width="240">
<img src="https://github.com/user-attachments/assets/7c1e55d7-9dc2-4df4-9166-d56b63b3d9cb" width="230">
<img src="https://github.com/user-attachments/assets/c57f70b5-80d0-49dd-b287-81334062b4aa" width="240">
</p>


## Gaussian truncated potential

The Coulombic splitting

```math
\dfrac{1}{r} = v_0(r) + v_1(r) \equiv \dfrac{\mathrm{ercf(\kappa r)}}{r} + \dfrac{\mathrm{erc(\kappa r)}}{r},
```
defines the short-ranged potential $v_0(r)$.

<div align="center">
  <img src="https://github.com/user-attachments/assets/5c85b7f2-4042-4bcd-b0d6-5e0452f68a2b" width="25%">
</div>

## Citation

The following papers illustrate the use of the code:

***A. T. Bui, S. J. Cox, **"Learning classical density functionals for ionic fluids"**, Phys. Rev. Lett. **134**, 148001 (2025)***

Links to: [arXiv:2410.02556](
https://doi.org/10.48550/arXiv.2410.02556) | [Phys. Rev. Lett.](https://doi.org/10.1103/PhysRevLett.134.148001)

***A. T. Bui, S. J. Cox, **"Dielectrocapillarity for exquisite control of fluids
"(2025)***

Links to: [arXiv:2503.09855](
https://doi.org/10.48550/arXiv.2503.09855)

## Installation

Requirements to run:
- Python >= 3.7
- NumPy
- PyYAML
- Scipy



Simply clone the repository:
   ```sh
   git clone https://github.com/annatbui/GCMC.git
   ```



## License

This code is licensed under the GNU License - see the LICENSE file for details.
