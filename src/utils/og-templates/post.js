import { SITE } from "@/config";
import { escapeXml, renderTextLines, wrapText } from "./shared";

export default async post => {
  const titleLines = wrapText(post.data.title, 24, 4);
  const author = post.data.author || SITE.author;

  return `
    <svg width="1200" height="630" viewBox="0 0 1200 630" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="1200" height="630" fill="#F8FAFC" />
      <rect x="86" y="76" width="1028" height="478" rx="28" fill="#E2E8F0" />
      <rect x="72" y="62" width="1028" height="478" rx="28" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="2" />
      <text x="132" y="152" font-size="22" font-weight="600" fill="#0369A1" letter-spacing="4">${escapeXml(SITE.title)}</text>
      ${renderTextLines(titleLines, 132, 248, 76, 60)}
      <text x="132" y="490" font-size="24" font-weight="500" fill="#475569">by ${escapeXml(author)}</text>
      <text x="1010" y="490" text-anchor="end" font-size="24" font-weight="600" fill="#0F172A">${escapeXml(new URL(SITE.website).hostname)}</text>
    </svg>
  `;
};
