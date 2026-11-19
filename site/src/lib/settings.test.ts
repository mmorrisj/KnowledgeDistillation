import { getJupyterUrl, labHref, normalizeJupyterUrl, setJupyterUrl } from "./settings";

describe("normalizeJupyterUrl", () => {
  it("strips trailing slashes and surrounding spaces", () => {
    expect(normalizeJupyterUrl("http://localhost:8888///")).toBe("http://localhost:8888");
    expect(normalizeJupyterUrl("  https://gpu.example.com/user/a/ ")).toBe("https://gpu.example.com/user/a");
  });
  it("rejects anything that is not http or https", () => {
    expect(normalizeJupyterUrl("javascript:alert(1)")).toBeNull();
    expect(normalizeJupyterUrl("localhost:8888")).toBeNull();
    expect(normalizeJupyterUrl("ftp://host/")).toBeNull();
    expect(normalizeJupyterUrl("")).toBeNull();
  });
  it("drops query strings and fragments, which often carry a login token", () => {
    expect(normalizeJupyterUrl("http://localhost:8888/lab?token=abc#x")).toBe("http://localhost:8888/lab");
  });
});

describe("labHref", () => {
  it("joins the base and an encoded notebook path", () => {
    expect(labHref("http://h:8888", "labs/01 intro/a.ipynb")).toBe(
      "http://h:8888/lab/tree/labs/01%20intro/a.ipynb",
    );
  });
});

describe("stored Jupyter address", () => {
  it("is null until set", () => {
    expect(getJupyterUrl()).toBeNull();
  });
  it("saves the normalized value", () => {
    expect(setJupyterUrl("http://localhost:8888/")).toBe(true);
    expect(getJupyterUrl()).toBe("http://localhost:8888");
  });
  it("refuses an invalid value and keeps the old one", () => {
    setJupyterUrl("http://localhost:8888");
    expect(setJupyterUrl("javascript:alert(1)")).toBe(false);
    expect(getJupyterUrl()).toBe("http://localhost:8888");
  });
  it("clears the saved value when given an empty string", () => {
    setJupyterUrl("http://localhost:8888");
    expect(setJupyterUrl("")).toBe(true);
    expect(getJupyterUrl()).toBeNull();
  });
  it("ignores a stored value that is no longer valid", () => {
    localStorage.setItem("kd-course:settings:v1", JSON.stringify({ jupyterUrl: "javascript:x" }));
    expect(getJupyterUrl()).toBeNull();
  });
});
