# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under Apache-2.0; see LICENSE.
# Release adaptation: expose only the ViT-S/14 constructor used by BladeGL.
from dinov2.hub.backbones import dinov2_vits14
dependencies = ["torch"]
