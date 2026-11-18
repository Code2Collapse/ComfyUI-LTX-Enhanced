// LTX-Enhanced settings (L3.02; design docs/research/L3_settings_secrets_audit.md S9 in the C2C workspace).
// The backend reads the saved values through _c2c_saved_setting.py. The environment variables LTXE_DISABLE,
// LTXE_DISABLE_HDR and LTXE_DISABLE_TILING still win when they are set.
import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "C2C.LTXE.Settings",
    settings: [
        {
            id: "c2c.ltxe.enabled",
            name: "Load the LTX-Enhanced nodes (restart to apply)",
            category: ["C2C", "LTX-Enhanced", "General", "enabled"],
            type: "boolean",
            defaultValue: true,
            tooltip: "Off: after the next restart the pack registers no nodes, and saved workflows show them as "
                + "missing instead of running something different. LTXE_DISABLE=1 does the same and wins.",
        },
        {
            id: "c2c.ltxe.hdr.enabled",
            name: "Load the SDR -> HDR helper nodes (restart to apply)",
            category: ["C2C", "LTX-Enhanced", "General", "hdr"],
            type: "boolean",
            defaultValue: true,
            tooltip: "Off: after the next restart only the HDR helper nodes are left out. "
                + "LTXE_DISABLE_HDR=1 does the same and wins.",
        },
        {
            id: "c2c.ltxe.tiling.enabled",
            name: "LTXE Tiled Model splits the model call into tiles",
            category: ["C2C", "LTX-Enhanced", "Memory", "tiling"],
            type: "boolean",
            defaultValue: true,
            tooltip: "Off: LTXE Tiled Model passes the model through unchanged (applies to the next run). "
                + "LTXE_DISABLE_TILING=1 does the same and wins.",
        },
    ],
});
