Troubleshooting: pygame warnings

If you see output like:

```
RuntimeWarning: Your system is avx2 capable but pygame was not built with support for it. The performance of some of your blits could be adversely affected. Consider enabling compile time detection with environment variables like PYGAME_DETECT_AVX2=1 if you are compiling without cross compilation.
UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30.
```

These are not fatal errors — the simulator should still run — but here are recommended responses:

- AVX2 / pygame build warning
  - Cause: prebuilt pygame wheels may not be compiled to take advantage of CPU SIMD instructions available on your machine. This only affects some blit performance (rendering speed).
  - Quick option: ignore it if the GUI is responsive enough for your needs.
  - Rebuild pygame with AVX2 detection (Windows PowerShell example, in your Python environment):

```powershell
$env:PYGAME_DETECT_AVX2 = '1'
python -m pip install --upgrade pip setuptools
python -m pip install --no-binary :all: pygame
```

  - Notes: Building from source requires a working C compiler and SDL dependencies. If you use conda, try installing pygame from conda-forge which often provides compatible builds:

```powershell
conda install -c conda-forge pygame
```

- pkg_resources / setuptools warning
  - Cause: some packages still import pkg_resources from setuptools; setuptools 81+ changes/removes that API and triggers the deprecation message.
  - Short-term fix: pin setuptools to a pre-removal release. This project includes `setuptools<81` in `requirements.txt` to avoid the warning when installing packages via pip.
  - Long-term: upgrade dependent packages (or their authors should update) to avoid relying on pkg_resources.

If you'd like, I can add more detailed build steps for Windows (MSVC) or add a `CONTRIBUTING.md` section with build notes for pygame.
