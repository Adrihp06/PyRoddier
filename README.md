# PyRoddier - Astronomical Image Analysis

This software is designed to detect optical aberrations in telescopes using the Roddier test.
It analyzes intra- and extrafocal images to compute the wavefront and decompose it into Zernike modes, helping users evaluate the optical quality of their instruments.

## License

This project is licensed under the [MIT License](./LICENSE).
You are free to use, modify, and distribute this software for both personal and commercial purposes, provided that the original license and copyright notice are included.


## Description

The project implements a set of tools for astronomical image analysis, including:

- Astronomical image processing
- Optical aberration analysis
- Zernike coefficients calculation
- Graphical interface for visualization and analysis

## Requirements

The project requires the following main dependencies:

- Python 3.11+
- PyQt5 >= 5.15.0
- NumPy >= 1.19.0
- Matplotlib >= 3.3.0
- Astropy >= 4.0.0
- SciPy >= 1.6.0
- Pillow >= 8.0.0

## Installation

1. Clone the repository:
```bash
git clone [REPOSITORY_URL]
cd PyRoddier
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
PyRoddier/
├── src/                # Main source code
│   ├── core/          # Core functionalities
│   ├── gui/           # Graphical interface
│   ├── commom/        # Utilities
├── tests/             # Unit tests
├── images/            # Example images
└── docs/              # Documentation
```

## Usage

To run the main application:

```bash
python src/main.py
```

## Main Features

### Image Analysis

The system implements algorithms for astronomical image analysis, including:

- Star detection
- FWHM measurement
- PSF analysis

### Zernike Coefficients Calculation

The system calculates Zernike coefficients to characterize optical aberrations:

$$
Z_n^m(\rho, \theta) = R_n^m(\rho) \cdot \cos(m\theta)
$$

where:
- $R_n^m(\rho)$ are the Zernike radial polynomials
- $n$ is the radial order
- $m$ is the azimuthal order

## Development

### Running Tests

To run the test suite:

```bash
python run_tests.py
```

### Code Coverage

To generate a coverage report:

```bash
coverage run -m pytest
coverage html
```

## Contributing

Contributions are welcome. Please ensure to:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

adrianhdezp10@gmail.com

Project Link: https://github.com/Adrihp06/PyRoddier
