import { SITE } from "@/config";
import { escapeXml, renderTextLines, wrapText } from "./shared";

export default async () => {
  const titleLines = wrapText(SITE.title, 12, 2);
  const descLines = wrapText(SITE.desc, 28, 3);
  const domain = new URL(SITE.website).hostname;

  return `
    <svg width="1200" height="630" viewBox="0 0 1200 630" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="1200" height="630" fill="#F8FAFC" />
      <rect x="86" y="76" width="1028" height="478" rx="28" fill="#E2E8F0" />
      <rect x="72" y="62" width="1028" height="478" rx="28" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="2" />
      <circle cx="132" cy="124" r="8" fill="#0F172A" opacity="0.2" />
      <circle cx="160" cy="124" r="8" fill="#0F172A" opacity="0.14" />
      <circle cx="188" cy="124" r="8" fill="#0F172A" opacity="0.1" />
      <text x="132" y="214" font-size="22" font-weight="600" fill="#0369A1" letter-spacing="4">FUCOV</text>
      ${renderTextLines(titleLines, 132, 308, 92, 76)}
      ${descLines
        .map(
          (line, index) =>
            `<text x="132" y="${388 + index * 40}" font-size="28" font-weight="500" fill="#475569">${escapeXml(line)}</text>`
        )
        .join("")}
      <text x="132" y="490" font-size="24" font-weight="600" fill="#0F172A">AI Infra / RecSys / Agents</text>
      <text x="1010" y="500" text-anchor="end" font-size="24" font-weight="600" fill="#0F172A">${escapeXml(domain)}</text>
    </svg>
  `;
};
