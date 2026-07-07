{
  description = "AI Swarm Intelligence — Evolutionary GPU Compute Sandbox";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          name = "swarm-evolution-sandbox";

          buildInputs = with pkgs; [
            python3
            git
            gh
            bc
            opencode
            expect
            util-linux
            rustc
            cargo
            gcc
            pkg-config
            libiconv
            python3Packages.psutil
            ocl-icd
            opencl-headers
            intel-compute-runtime-legacy1
            clinfo
          ];

          shellHook = ''
            export OCL_ICD_VENDORS="${pkgs.intel-compute-runtime-legacy1}/etc/OpenCL/vendors"
            export LD_LIBRARY_PATH="${pkgs.ocl-icd}/lib:${pkgs.intel-compute-runtime-legacy1}/lib:$LD_LIBRARY_PATH"
            echo "[swarm] Polyglot evolution sandbox ready — $(python3 --version)"
          '';
        };
      });
}
