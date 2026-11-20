import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import NeighbourOverlapExplorer from "./NeighbourOverlapExplorer";
import VectorSizeExplorer from "./VectorSizeExplorer";

describe("VectorSizeExplorer", () => {
  it("shows the storage for the starting choice", () => {
    render(<VectorSizeExplorer />);
    expect(screen.getByRole("status")).toHaveTextContent("3.8 GB");
  });
  it("shrinks when a smaller embedding size is chosen", async () => {
    render(<VectorSizeExplorer />);
    await userEvent.click(screen.getByRole("button", { name: "384" }));
    expect(screen.getByRole("status")).toHaveTextContent("1.4 GB");
  });
  it("grows with the number of texts", async () => {
    render(<VectorSizeExplorer />);
    await userEvent.click(screen.getByRole("button", { name: "100 million" }));
    expect(screen.getByRole("status")).toHaveTextContent("381 GB");
  });
});

describe("NeighbourOverlapExplorer", () => {
  it("reports the overlap for the chosen number of results", () => {
    render(<NeighbourOverlapExplorer />);
    fireEvent.change(screen.getByRole("slider", { name: /results compared/i }), { target: { value: "1" } });
    expect(screen.getByRole("status")).toHaveTextContent("100%");
  });
  it("marks shared results in words, not only by color", () => {
    render(<NeighbourOverlapExplorer />);
    expect(screen.getAllByText(/in both/i).length).toBeGreaterThan(0);
  });
});
