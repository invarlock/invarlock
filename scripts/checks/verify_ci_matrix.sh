#!/bin/bash
# InvarLock CI Matrix Verification Script
# Verifies that CI matrix configurations are valid

set -e

echo "🧪 Verifying CI matrix configurations..."

# Check that preset configs exist
echo "📋 Checking preset and workflow configurations..."
for config in \
    configs/presets/causal_lm/wikitext2_512.yaml \
    configs/overlays/edits/quant_rtn/8bit_attn.yaml; do
    if [ -f "$config" ]; then
        echo "  ✅ $config"
    else
        echo "  ❌ $config missing"
        exit 1
    fi
done

# Validate YAML syntax
echo "📝 Validating YAML syntax..."
if command -v python3 &> /dev/null; then
    if python3 -c "import yaml" &> /dev/null; then
        for config in \
            configs/presets/**/*.yaml \
            configs/overlays/**/*.yaml \
            configs/calibration/*.yaml \
            configs/overrides/*.yaml \
            src/invarlock/_data/runtime/tiers.yaml \
            src/invarlock/_data/runtime/profiles/*.yaml; do
            if [ -f "$config" ]; then
                python3 -c "import yaml; yaml.safe_load(open('$config'))" || {
                    echo "❌ Invalid YAML: $config"
                    exit 1
                }
                echo "  ✅ $config"
            fi
        done
    else
        echo "  ⚠️  PyYAML not installed, skipping YAML validation"
    fi
else
    echo "  ⚠️  Python3 not available, skipping YAML validation"
fi

# Core edit availability check reduced to quant only
echo "🔧 Checking core edit availability..."
if command -v rg >/dev/null 2>&1; then
    EDIT_CHECK='rg -n "class RTNQuantEdit" src/invarlock/edits/quant_rtn.py'
else
    EDIT_CHECK='grep -n "class RTNQuantEdit" src/invarlock/edits/quant_rtn.py'
fi

if eval "$EDIT_CHECK" >/dev/null 2>&1; then
    echo "  ✅ quant_rtn"
else
    echo "  ❌ quant_rtn missing"
    exit 1
fi

# Check CI workflow exists
echo "🚀 Checking CI workflow..."
if [ -f ".github/workflows/ci.yml" ]; then
    echo "  ✅ CI workflow exists"
else
    echo "  ❌ CI workflow missing"
    exit 1
fi

echo "✅ CI matrix verification completed successfully"
