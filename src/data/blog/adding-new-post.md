---
author: Sat Naing
pubDatetime: 2022-09-23T15:22:00Z
modDatetime: 2025-06-13T16:52:45.934Z
title: 在 AstroPaper 主题中添加新文章
slug: adding-new-posts-in-astropaper-theme
featured: true
draft: false
tags:
  - docs
description: 在 AstroPaper 主题中创建或添加新文章时的一些规则与建议。
---

这篇文章总结了在 AstroPaper 博客主题中创建新文章时的一些规则、建议，以及实用的小技巧。

<figure>
  <img
    src="https://images.pexels.com/photos/159618/still-life-school-retro-ink-159618.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"
    alt="经典木质书桌、书写工具、复古时钟与皮包的照片"
  />
  <figcaption class="text-center">
    图片来源：<a href="https://www.pexels.com/photo/brown-wooden-desk-159618/">Pixabay</a>
  </figcaption>
</figure>

## Table of contents

## 创建博客文章

要编写一篇新的博客文章，请在 `src/data/blog/` 目录下创建一个 Markdown 文件。

> 在 AstroPaper v5.1.0 之前，所有博客文章都必须放在 `src/data/blog/` 目录中，不能进一步按子目录组织。

从 AstroPaper v5.1.0 开始，你可以将博客文章放入子目录中，管理内容会更方便。

例如，如果你想把文章按 `2025` 分类存放，可以放到 `src/data/blog/2025/` 下。这样也会影响文章 URL，因此 `src/data/blog/2025/example-post.md` 对应的访问地址会变成 `/posts/2025/example-post`。

如果你**不希望子目录影响文章 URL**，只需要在目录名前加一个下划线 `_` 即可。

```bash
# 示例：博客文章目录结构与对应 URL
src/data/blog/very-first-post.md          -> mysite.com/posts/very-first-post
src/data/blog/2025/example-post.md        -> mysite.com/posts/2025/example-post
src/data/blog/_2026/another-post.md       -> mysite.com/posts/another-post
src/data/blog/docs/_legacy/how-to.md      -> mysite.com/posts/docs/how-to
src/data/blog/Example Dir/Dummy Post.md   -> mysite.com/posts/example-dir/dummy-post
````

> 💡 提示：你也可以在 frontmatter 中手动覆盖文章的 slug。下面一节会详细说明。

如果你发现构建结果里没有出现对应的子目录 URL，可以尝试删除 `node_modules`，重新安装依赖后再重新构建。

## Frontmatter

Frontmatter 是存放文章关键信息的主要位置。它位于文章顶部，使用 YAML 格式编写。关于 frontmatter 的更多说明，可以参考 [Astro 官方文档](https://docs.astro.build/en/guides/markdown-content/)。

下面是每篇文章可用的 frontmatter 字段说明。

| 属性                 | 说明                                          | 备注                                     |
| ------------------ | ------------------------------------------- | -------------------------------------- |
| ***title***        | 文章标题（h1）                                    | 必填<sup>*</sup>                         |
| ***description***  | 文章描述，用于文章摘要和页面描述                            | 必填<sup>*</sup>                         |
| ***pubDatetime***  | 发布时间，使用 ISO 8601 格式                         | 必填<sup>*</sup>                         |
| ***modDatetime***  | 修改时间，使用 ISO 8601 格式（仅在文章修改后再添加）             | 可选                                     |
| ***author***       | 文章作者                                        | 默认值为 `SITE.author`                     |
| ***slug***         | 文章的 slug，可选                                 | 默认值为根据文件名自动生成的 slug                    |
| ***featured***     | 是否在首页的精选文章区域中展示                             | 默认值为 `false`                           |
| ***draft***        | 是否将文章标记为未发布                                 | 默认值为 `false`                           |
| ***tags***         | 文章相关标签，使用 YAML 数组格式书写                       | 默认值为 `others`                          |
| ***ogImage***      | 文章的 OG 图片，用于社交分享和 SEO，可填写远程 URL 或当前目录相对路径   | 默认值为 `SITE.ogImage` 或自动生成的 OG 图片       |
| ***canonicalURL*** | 规范链接（绝对地址），适用于文章已在其他站点发布的情况                 | 默认值为 `Astro.site + Astro.url.pathname` |
| ***hideEditPost*** | 是否隐藏文章标题下方的“编辑此页”按钮，仅对当前文章生效                | 默认值为 `false`                           |
| ***timezone***     | 为当前文章单独指定 IANA 时区，会覆盖全局的 `SITE.timezone` 设置 | 默认值为 `SITE.timezone`                   |

> 提示：你可以在控制台运行 `new Date().toISOString()` 来快速获取 ISO 8601 格式的时间。注意把两边的引号去掉。

其中，frontmatter 中**必须填写**的只有 `title`、`description` 和 `pubDatetime`。

标题和描述（摘要）对搜索引擎优化（SEO）非常重要，因此 AstroPaper 推荐你在每篇文章中都认真填写这两个字段。

`slug` 是 URL 的唯一标识，因此必须保证它在所有文章中唯一，不能与其他文章重复。slug 中的空格应使用 `-` 或 `_` 分隔，推荐使用 `-`。默认情况下，slug 会根据文章文件名自动生成；当然，你也可以在 frontmatter 中手动指定它。

例如，如果文章文件名是 `adding-new-post.md`，而你没有在 frontmatter 中指定 `slug`，那么 Astro 会自动根据文件名生成 slug，也就是 `adding-new-post`。如果你在 frontmatter 中显式写了 `slug`，则会覆盖默认值。更多说明可以参考 [Astro 官方文档](https://docs.astro.build/en/guides/content-collections/#defining-custom-slugs)。

如果你在某篇文章中省略了 `tags` 字段，也就是没有指定任何标签，那么系统会自动使用默认标签 `others`。你可以在 `content.config.ts` 中修改这个默认值。

```ts file="src/content.config.ts"
export const blogSchema = z.object({
  // ...
  draft: z.boolean().optional(),
  // [!code highlight:1]
  tags: z.array(z.string()).default(["others"]), // 可将 "others" 改成你想要的默认标签
  // ...
});
```

### Frontmatter 示例

下面是一篇文章的 frontmatter 示例。

```yaml file="src/data/blog/sample-post.md"
---
title: 文章标题
author: 你的名字
pubDatetime: 2022-09-21T05:17:19Z
slug: the-title-of-the-post
featured: true
draft: false
tags:
  - some
  - example
  - tags
ogImage: ../../assets/images/example.png # 对应 src/assets/images/example.png
# ogImage: "https://example.org/remote-image.png" # 远程图片 URL
description: 这是这篇示例文章的描述。
canonicalURL: https://example.org/my-article-was-already-posted-here
---
```

## 添加目录

默认情况下，文章页面**不会自动显示目录（toc）**。如果你希望启用目录，需要按特定方式书写。

你需要使用二级标题格式写出 `Table of contents`，并将它放在你希望目录出现的位置。

例如，如果你想把目录放在引言段落下方（这也是我平时最常用的方式），可以这样写：

<!-- prettier-ignore-start -->

```md
---
# frontmatter
---

这里是一些关于在 AstroPaper 博客主题中创建新文章的建议、技巧和注意事项。

<!-- [!code ++] -->
## Table of contents

<!-- 下面是文章的其余内容 -->
```

<!-- prettier-ignore-end -->

## 标题层级

关于标题层级，还有一点需要注意。AstroPaper 会把 frontmatter 中的 `title` 作为整篇文章的主标题，因此正文中的其他标题应从 h2 到 h6 来使用。

这条规则不是强制的，但从视觉层级、可访问性和 SEO 的角度来看，都非常推荐遵循。

## 代码高亮

AstroPaper 默认使用 [Shiki](https://shiki.style/) 来实现代码高亮。从 AstroPaper v5.4 开始，又引入了 [@shikijs/transformers](https://shiki.style/packages/transformers) 来增强围栏代码块的表现效果。如果你不想使用这些增强能力，也可以直接将其移除。

```bash
pnpm remove @shikijs/transformers
```

```js file="astro.config.ts"
// ...
// [!code --:5]
import {
  transformerNotationDiff,
  transformerNotationHighlight,
  transformerNotationWordHighlight,
} from "@shikijs/transformers";

export default defineConfig({
  // ...
  markdown: {
    remarkPlugins: [remarkToc, [remarkCollapse, { test: "Table of contents" }]],
    shikiConfig: {
      // 更多主题可参考 https://shiki.style/themes
      themes: { light: "min-light", dark: "night-owl" },
      defaultColor: false,
      wrap: false,
      transformers: [
        transformerFileName(),
      // [!code --:3]
        transformerNotationHighlight(),
        transformerNotationWordHighlight(),
        transformerNotationDiff({ matchAlgorithm: "v3" }),
      ],
    },
  },
  // ...
}
```

## 为博客内容存储图片

在 Markdown 文件中插入图片，常见有两种方式。

> 注意：如果你需要在 Markdown 中对优化后的图片进行额外样式控制，建议使用 [MDX](https://docs.astro.build/en/guides/images/#images-in-mdx-files)。

### 存放在 `src/assets/` 目录中（推荐）

你可以把图片放在 `src/assets/` 目录中。这些图片会通过 Astro 的 [Image Service API](https://docs.astro.build/en/reference/image-service-reference/) 自动进行优化。

引用时可以使用相对路径，也可以使用别名路径（`@/assets/`）。

例如，假设你想显示的 `example.jpg` 路径是 `/src/assets/images/example.jpg`。

```md
![something](@/assets/images/example.jpg)

<!-- 或者 -->

![something](../../assets/images/example.jpg)

<!-- 使用 img 标签或 Image 组件都不适用于这种写法 ❌ -->
<img src="@/assets/images/example.jpg" alt="something">
<!-- ^^ 这是错误写法 -->
```

> 从技术上说，你也可以把图片放在 `src` 下的任意目录中。这里推荐 `src/assets` 只是为了更清晰地组织资源。

### 存放在 `public` 目录中

你也可以把图片放到 `public` 目录中。但要注意，`public` 下的图片不会经过 Astro 的处理，因此不会自动优化，压缩和优化工作需要你自己完成。

引用这类图片时，应使用绝对路径。它们既可以通过 [Markdown 图片语法](https://www.markdownguide.org/basic-syntax/#images-1) 展示，也可以直接通过 [HTML 的 img 标签](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/img) 展示。

例如，假设 `example.jpg` 位于 `/public/assets/images/example.jpg`。

```md
![something](/assets/images/example.jpg)

<!-- 或者 -->

<img src="/assets/images/example.jpg" alt="something">
```

## 补充内容

### 图片压缩

当你在文章中插入图片时，尤其是放在 `public` 目录下的图片，建议提前进行压缩。这会直接影响网站整体性能。

这里推荐两个常用的在线压缩工具：

* [TinyPng](https://tinypng.com/)
* [TinyJPG](https://tinyjpg.com/)

### OG 图片

如果文章没有显式指定 OG 图片，那么系统会使用默认的 OG 图片。虽然不是必须项，但仍然建议你为每篇文章设置与内容相关的 OG 图片。推荐尺寸是 **1200 × 640 px**。

> 自 AstroPaper v1.4.0 起，如果未指定 OG 图片，系统会自动生成。可查看这篇公告：[动态生成 AstroPaper 博客文章的 OG 图片](https://astro-paper.pages.dev/posts/dynamic-og-image-generation-in-astropaper-blog-posts/)。

