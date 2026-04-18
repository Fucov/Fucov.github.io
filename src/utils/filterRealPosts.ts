import { existsSync } from "node:fs";
import { resolve } from "node:path";
import type { CollectionEntry } from "astro:content";

const filterRealPosts = (posts: CollectionEntry<"blog">[]) =>
  posts.filter(post => {
    if (!post.filePath) return false;
    return existsSync(resolve(post.filePath));
  });

export default filterRealPosts;
