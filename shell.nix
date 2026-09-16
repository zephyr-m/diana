{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  packages = with pkgs; [ python312 uv git pkg-config ];
  # Shared libraries needed by prebuilt Python audio/ONNX wheels on NixOS.
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
    stdenv.cc.cc.lib zlib libsndfile glib libGL
  ]);
  shellHook = ''
    export UV_PYTHON="${pkgs.python312}/bin/python3.12"
    export UV_PYTHON_DOWNLOADS=never
  '';
}
