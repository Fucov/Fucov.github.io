# Fucov Blog

An AstroPaper-based personal blog for Fucov.

## Overview

- Keeps the AstroPaper reading-first layout and dark mode.
- Adds a homepage intro and featured project cards.
- Replaces template author information with Fucov's profile.
- Supports optional Giscus comments on post detail pages only.
- Analytics is disabled by default and optional.
- Removes runtime dependence on remote Google font metadata for local dev.

## Local Development

```bash
pnpm install
pnpm dev
```

Build for production:

```bash
pnpm build
```

## Environment Variables

Copy `.env.example` to `.env` and fill only what you need.

### Giscus

Enable comments only after all required values are configured:

```bash
PUBLIC_GISCUS_ENABLED=true
PUBLIC_GISCUS_REPO=Fucov/Fucov.github.io
PUBLIC_GISCUS_REPO_ID=
PUBLIC_GISCUS_CATEGORY=Announcements
PUBLIC_GISCUS_CATEGORY_ID=
PUBLIC_GISCUS_MAPPING=pathname
PUBLIC_GISCUS_STRICT=0
PUBLIC_GISCUS_REACTIONS_ENABLED=1
PUBLIC_GISCUS_EMIT_METADATA=0
PUBLIC_GISCUS_INPUT_POSITION=bottom
PUBLIC_GISCUS_THEME=preferred_color_scheme
PUBLIC_GISCUS_LANG=zh-CN
```

If any required Giscus value is missing, the blog will not render the comment section and the page will still work normally.

## Required Manual Setup

### Giscus GitHub Setup

1. Install the [Giscus GitHub App](https://github.com/apps/giscus) on `Fucov/Fucov.github.io`.
2. Enable GitHub Discussions for the repository.
3. Create or choose a discussion category, for example `Announcements`.
4. Copy the generated `repoId` and `categoryId` from the Giscus configuration page.
5. Put those values into `.env`.

## Content And Images

- Homepage hero and featured project covers use Astro's native image pipeline.
- Existing markdown posts can be added under `src/data/blog/`.
- For future posts, prefer importing local images from `src/assets/images` and using Astro image syntax where practical. That gives better dimensions, smaller layout shift, and responsive output.

## Markdown And LaTeX

The blog now supports standard Astro Markdown plus LaTeX math rendering through `remark-math` and `rehype-katex`.

Inline math:

```md
The famous equation is $E = mc^2$.
```

Block math:

```md
$$
\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
$$
```

Notes:

- KaTeX styles are loaded locally from the installed `katex` package, not from a CDN.
- Math works directly inside Markdown posts under `src/data/blog/`.
- Existing Markdown features such as tables, blockquotes, code blocks, figures, and footnotes continue to work with the site typography styles.

## Main Files

- `src/config.ts`: site identity, SEO basics, timezone, edit URL.
- `src/constants.ts`: social links used by shared components.
- `src/pages/index.astro`: homepage structure.
- `src/components/HomeIntro.astro`: personal intro section.
- `src/components/FeaturedProjects.astro`: project cards above the post list.
- `src/components/Comments.astro`: optional Giscus integration.
- `src/components/Analytics.astro`: optional analytics injection.
- `src/layouts/Layout.astro`: global layout, metadata, theme bootstrap, analytics mount.
- `src/layouts/PostDetails.astro`: comments mounted only on post detail pages.
- `astro.config.ts`: removes remote font provider dependency while staying on Astro 5.

## Notes

- Current Astro version stays on Astro 5. No major upgrade was introduced.
- The site uses a system font stack instead of remote Google font metadata.
- Dynamic OG image generation remains available without remote font fetching.
