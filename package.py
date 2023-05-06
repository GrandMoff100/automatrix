"""A module for packaging the automatrix package."""
import sys
import zipfile

with zipfile.PyZipFile(
    sys.argv[1],
    mode="w",
    optimize=2,
    compression=zipfile.ZIP_DEFLATED,
) as zip_app:
    zip_app.writepy("automatrix", filterfunc=lambda file: file.endswith(".pyc") or file != "__main__.py")
    zip_app.writepy("automatrix/__main__.py")
