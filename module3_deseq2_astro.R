# ============================================================
# BST 281 Final Project — Module 3
# Astrocyte pseudo-bulk DESeq2 analysis
# ============================================================

suppressPackageStartupMessages({
  library(DESeq2)
  library(tidyverse)
  library(pheatmap)
  library(ggrepel)
})

# -----------------------------
# Paths
# -----------------------------

outdir <- "module3_astro"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

counts_path <- file.path(outdir, "astro_pseudobulk_counts_genes_by_donor.csv")
meta_path   <- file.path(outdir, "astro_sample_metadata.csv")

# -----------------------------
# Load pseudo-bulk counts
# -----------------------------

counts <- read.csv(counts_path, row.names = 1, check.names = FALSE)
meta <- read.csv(meta_path, row.names = 1, check.names = FALSE)

# Make sure sample order matches
counts <- counts[, rownames(meta)]

# Convert to integer count matrix
counts <- round(as.matrix(counts))
mode(counts) <- "integer"

# Factor variables
meta$disease <- factor(meta$disease, levels = c("Control", "AD"))
meta$sex <- factor(meta$sex)

print(meta)
print(dim(counts))

# -----------------------------
# Filter low-count genes
# -----------------------------

keep <- rowSums(counts) >= 10 & rowSums(counts > 0) >= 3
counts_filt <- counts[keep, ]

cat("Genes before filtering:", nrow(counts), "\n")
cat("Genes after filtering:", nrow(counts_filt), "\n")

# -----------------------------
# DESeq2 model
# -----------------------------

dds <- DESeqDataSetFromMatrix(
  countData = counts_filt,
  colData = meta,
  design = ~ sex + disease
)

dds <- DESeq(dds)

res <- results(
  dds,
  contrast = c("disease", "AD", "Control")
)

res_df <- as.data.frame(res) %>%
  rownames_to_column("gene") %>%
  arrange(pvalue)

write.csv(
  res_df,
  file.path(outdir, "astro_DESeq2_AD_vs_Control_results.csv"),
  row.names = FALSE
)

cat("Top DESeq2 results:\n")
print(head(res_df, 20))

# -----------------------------
# Variance-stabilizing transform
# -----------------------------

vsd <- vst(dds, blind = FALSE)
vsd_mat <- assay(vsd)

write.csv(
  vsd_mat,
  file.path(outdir, "astro_vst_expression_matrix.csv")
)

# -----------------------------
# PCA plot
# -----------------------------

pca_data <- plotPCA(vsd, intgroup = c("disease", "sex"), returnData = TRUE)
percent_var <- round(100 * attr(pca_data, "percentVar"))

p_pca <- ggplot(pca_data, aes(PC1, PC2, color = disease, shape = sex, label = name)) +
  geom_point(size = 4) +
  geom_text_repel(size = 3) +
  xlab(paste0("PC1: ", percent_var[1], "% variance")) +
  ylab(paste0("PC2: ", percent_var[2], "% variance")) +
  theme_bw() +
  ggtitle("Astrocyte pseudo-bulk PCA")

ggsave(
  file.path(outdir, "astro_pseudobulk_PCA.png"),
  p_pca,
  width = 7,
  height = 5,
  dpi = 300
)

# -----------------------------
# Volcano plot
# -----------------------------

volcano_df <- res_df %>%
  mutate(
    neg_log10_padj = -log10(padj),
    significant = ifelse(!is.na(padj) & padj < 0.05, "FDR < 0.05", "Not FDR < 0.05")
  )

top_labels <- volcano_df %>%
  filter(!is.na(pvalue)) %>%
  arrange(pvalue) %>%
  slice_head(n = 10)

p_volcano <- ggplot(volcano_df, aes(x = log2FoldChange, y = -log10(pvalue))) +
  geom_point(aes(color = significant), alpha = 0.7, size = 1.5) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  geom_text_repel(
    data = top_labels,
    aes(label = gene),
    size = 3,
    max.overlaps = 20
  ) +
  theme_bw() +
  labs(
    title = "Astrocyte pseudo-bulk DESeq2: AD vs Control",
    x = "log2 fold change: AD vs Control",
    y = "-log10 p-value"
  )

ggsave(
  file.path(outdir, "astro_DESeq2_volcano.png"),
  p_volcano,
  width = 7,
  height = 5,
  dpi = 300
)

# -----------------------------
# Heatmap of top DE genes
# -----------------------------

top_genes <- res_df %>%
  filter(!is.na(pvalue)) %>%
  arrange(pvalue) %>%
  slice_head(n = 30) %>%
  pull(gene)

heatmap_mat <- vsd_mat[top_genes, ]

# Z-score by gene
heatmap_mat_z <- t(scale(t(heatmap_mat)))

annotation_col <- meta[, c("disease", "sex", "n_nuclei"), drop = FALSE]

png(
  file.path(outdir, "astro_top30_DE_genes_heatmap.png"),
  width = 1000,
  height = 800,
  res = 120
)

pheatmap(
  heatmap_mat_z,
  annotation_col = annotation_col,
  cluster_rows = TRUE,
  cluster_cols = TRUE,
  fontsize_row = 8,
  main = "Top 30 astrocyte pseudo-bulk DE genes"
)

dev.off()

# -----------------------------
# Ranked gene list for enrichment
# -----------------------------

ranked_genes <- res_df %>%
  filter(!is.na(stat)) %>%
  arrange(desc(stat)) %>%
  dplyr::select(gene, stat, log2FoldChange, pvalue, padj)

write.csv(
  ranked_genes,
  file.path(outdir, "astro_ranked_genes_for_enrichment.csv"),
  row.names = FALSE
)

cat("Module 3 DESeq2 analysis complete.\n")

