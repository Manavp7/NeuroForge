// Lightweight Plotly wrapper: use the slim dist build via the factory to avoid
// pulling in the full plotly.js bundle.
import createPlotlyComponent from "react-plotly.js/factory";
// @ts-expect-error - dist-min has no bundled types
import Plotly from "plotly.js-dist-min";

const Plot = createPlotlyComponent(Plotly);
export default Plot;
