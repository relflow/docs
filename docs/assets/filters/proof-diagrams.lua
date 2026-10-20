function Image(image)
  local source = quarto.doc.input_file:gsub("\\", "/")
  local is_proof = source:match("^proofs/") or source:match("/proofs/")
  local is_benchmark = source:match("^benchmarks/") or source:match("/benchmarks/")
  if not (is_proof or is_benchmark) then
    return
  end
  if not quarto.doc.is_format("html") or not image.src:match("assets/diagrams/.+%.svg$") then
    return
  end
  return pandoc.Span({image}, pandoc.Attr("", {"proof-diagram"}, {
    tabindex = "0",
    role = "region",
    ["aria-label"] = is_benchmark and "Scrollable benchmark schema diagram" or "Scrollable proof diagram"
  }))
end
