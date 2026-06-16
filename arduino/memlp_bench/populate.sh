#!/usr/bin/env bash
# Populate this sketch with the memlp sources it needs.
#
# arduino-cli/IDE compiles every source in the sketch folder, so the library
# headers + .cpp must physically live here. The single source of truth is the
# memlp submodule (../../memlp); these copies are gitignored. Re-run after
# updating the submodule.
set -e
here="$(cd "$(dirname "$0")" && pwd)"
src="$here/../../memlp"
mkdir -p "$here/utils"
cp "$src"/MLP.h "$src"/MLP.cpp "$src"/Layer.h "$src"/Loss.h \
   "$src"/Utils.h "$src"/Utils.cpp "$src"/Sample.h \
   "$src"/StaticLayer.h "$src"/StaticMLP.h "$here/"
cp "$src"/utils/Serialise.hpp "$here/utils/"
echo "populated $here from $src"
