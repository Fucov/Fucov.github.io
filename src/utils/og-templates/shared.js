export function escapeXml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&apos;");
}

export function wrapText(text, maxCharsPerLine, maxLines) {
  const normalized = text.replace(/\s+/g, " ").trim();
  if (!normalized) return [""];

  const words = normalized.split(" ");
  const lines = [];
  let currentLine = "";

  for (const rawWord of words) {
    const segments = [];

    if (rawWord.length > maxCharsPerLine) {
      for (let index = 0; index < rawWord.length; index += maxCharsPerLine) {
        segments.push(rawWord.slice(index, index + maxCharsPerLine));
      }
    } else {
      segments.push(rawWord);
    }

    for (const word of segments) {
      const nextLine = currentLine ? `${currentLine} ${word}` : word;
      if (nextLine.length <= maxCharsPerLine) {
        currentLine = nextLine;
      } else {
        if (currentLine) lines.push(currentLine);
        currentLine = word;
      }

      if (lines.length === maxLines) break;
    }

    if (lines.length === maxLines) break;
  }

  if (lines.length < maxLines && currentLine) {
    lines.push(currentLine);
  }

  if (normalized.length > lines.join(" ").length) {
    const trimmed = lines.slice(0, maxLines);
    const lastLine = trimmed[trimmed.length - 1] ?? "";
    trimmed[trimmed.length - 1] =
      lastLine.length >= maxCharsPerLine
        ? `${lastLine.slice(0, maxCharsPerLine - 1)}…`
        : `${lastLine}…`;
    return trimmed;
  }

  return lines.slice(0, maxLines);
}

export function renderTextLines(lines, startX, startY, lineHeight, fontSize) {
  return lines
    .map(
      (line, index) =>
        `<text x="${startX}" y="${startY + index * lineHeight}" font-size="${fontSize}" font-weight="700" fill="#111827">${escapeXml(line)}</text>`
    )
    .join("");
}
