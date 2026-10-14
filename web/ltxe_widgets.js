/**
 * LTXE Resolution Plan / LTXE Pad Frames - show a widget only when it does something (ledger L7.112).
 *
 * custom_width / custom_height matter only for target "custom"; pad_value only for pad_mode "constant". Hidden
 * widgets keep their values and are still saved with the workflow. Works in the classic canvas and in Nodes 2.0
 * (the shared helper toggles both the LiteGraph layout and the DOM), and inside subgraphs (it acts on the node).
 */
import { app } from "../../scripts/app.js";
import { setWidgetVisible, vueFitNodeHeight, vueSyncNodeWidgets } from "./_widget_visibility.js";

const NODE_IDS = new Set(["LTXE_ResolutionPlan", "LTXE_PadFrames"]);
const RULES = [
  { control: "target", show: (v) => v === "custom", widgets: ["custom_width", "custom_height"] },
  { control: "pad_mode", show: (v) => v === "constant", widgets: ["pad_value"] },
];

const byName = (node, name) => node.widgets?.find((w) => w.name === name);

function apply(node) {
  let changed = false;
  for (const rule of RULES) {
    const control = byName(node, rule.control);
    if (!control) continue;
    const visible = rule.show(String(control.value));
    for (const name of rule.widgets) {
      const w = byName(node, name);
      if (!w) continue;
      if ((w.__mec_hidden === true) === visible) {
        setWidgetVisible(w, visible);
        changed = true;
      }
    }
  }
  if (!changed) return;
  vueSyncNodeWidgets(node);
  const sz = node.computeSize?.();
  if (sz) node.setSize?.([Math.max(node.size?.[0] ?? 0, sz[0]), sz[1]]);
  vueFitNodeHeight(node);
  node.setDirtyCanvas?.(true, true);
}

function hook(node) {
  if (node.__ltxeWidgets) return;
  node.__ltxeWidgets = true;
  for (const rule of RULES) {
    const control = byName(node, rule.control);
    if (!control) continue;
    const prev = control.callback;
    control.callback = function (...args) {
      const r = prev?.apply(this, args);
      try { apply(node); } catch (_e) { /* never take the node down */ }
      return r;
    };
  }
  apply(node);
}

app.registerExtension({
  name: "C2C.LTXEnhanced.Widgets",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODE_IDS.has(nodeData?.name)) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      try { hook(this); } catch (_e) { /* never take the node down */ }
      return r;
    };

    // a loaded workflow sets the saved values after creation: re-apply once they are in
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      try { hook(this); apply(this); } catch (_e) { /* ignore */ }
      return r;
    };
  },
});
