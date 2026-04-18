interface Window {
  __giscusHandlersBound?: boolean;
  theme?: {
    themeValue: string;
    setPreference: () => void;
    reflectPreference: () => void;
    getTheme: () => string;
    setTheme: (val: string) => void;
  };
}

interface ImportMetaEnv {
  readonly PUBLIC_GOOGLE_SITE_VERIFICATION?: string;
  readonly PUBLIC_GISCUS_ENABLED?: string;
  readonly PUBLIC_GISCUS_REPO?: string;
  readonly PUBLIC_GISCUS_REPO_ID?: string;
  readonly PUBLIC_GISCUS_CATEGORY?: string;
  readonly PUBLIC_GISCUS_CATEGORY_ID?: string;
  readonly PUBLIC_GISCUS_MAPPING?: string;
  readonly PUBLIC_GISCUS_STRICT?: string;
  readonly PUBLIC_GISCUS_REACTIONS_ENABLED?: string;
  readonly PUBLIC_GISCUS_EMIT_METADATA?: string;
  readonly PUBLIC_GISCUS_INPUT_POSITION?: string;
  readonly PUBLIC_GISCUS_THEME?: string;
  readonly PUBLIC_GISCUS_LANG?: string;
  readonly PUBLIC_ANALYTICS_PROVIDER?: string;
  readonly PUBLIC_UMAMI_SCRIPT_SRC?: string;
  readonly PUBLIC_UMAMI_WEBSITE_ID?: string;
  readonly PUBLIC_PLAUSIBLE_SCRIPT_SRC?: string;
  readonly PUBLIC_PLAUSIBLE_DOMAIN?: string;
  readonly PUBLIC_GA_MEASUREMENT_ID?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
