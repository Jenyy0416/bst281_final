# ============================================================
# BST 281 Final Project — Module 3 Analysis 2
# Functional enrichment analysis for astrocyte pseudo-bulk DESeq2
# ============================================================
suppressPackageStartupMessages({
  library(tidyverse)
  library(clusterProfiler)
  library(org.Hs.eg.db)
  library(msigdbr)
  library(enrichplot)
  library(ggplot2)
})

# ============================================================
# 1. Paths
# ============================================================

outdir <- "module3_astro"
enrich_outdir <- file.path(outdir, "enrichment")
dir.create(enrich_outdir, recursive = TRUE, showWarnings = FALSE)

deseq_path <- file.path(outdir, "astro_DESeq2_AD_vs_Control_results.csv")
ranked_path <- file.path(outdir, "astro_ranked_genes_for_enrichment.csv")

# ============================================================
# 2. Load DESeq2 results
# ============================================================

res_df <- read.csv(deseq_path, stringsAsFactors = FALSE)
ranked_df <- read.csv(ranked_path, stringsAsFactors = FALSE)

cat("DESeq2 results loaded:\n")
cat("Number of genes:", nrow(res_df), "\n")
print(head(res_df))

# Make sure required columns exist
required_cols <- c("gene", "log2FoldChange", "stat", "pvalue", "padj")
missing_cols <- setdiff(required_cols, colnames(res_df))

if (length(missing_cols) > 0) {
  stop("Missing required columns in DESeq2 results: ",
       paste(missing_cols, collapse = ", "))
}

# ============================================================
# 3. Prepare gene IDs
# ============================================================

# Convert gene symbols to Entrez IDs
gene_map <- bitr(
  res_df$gene,
  fromType = "SYMBOL",
  toType = c("ENTREZID", "SYMBOL"),
  OrgDb = org.Hs.eg.db
)

cat("Successfully mapped genes:", nrow(gene_map), "\n")

res_mapped <- res_df %>%
  inner_join(gene_map, by = c("gene" = "SYMBOL"))

# Remove duplicated Entrez IDs by keeping the gene with the strongest absolute DESeq2 statistic
res_mapped_unique <- res_mapped %>%
  filter(!is.na(stat)) %>%
  arrange(desc(abs(stat))) %>%
  distinct(ENTREZID, .keep_all = TRUE)

# ============================================================
# 4. Prepare ranked gene list for GSEA
# ============================================================

gene_list <- res_mapped_unique$stat
names(gene_list) <- res_mapped_unique$ENTREZID

# Sort decreasing for GSEA
gene_list <- sort(gene_list, decreasing = TRUE)

cat("Ranked gene list length:", length(gene_list), "\n")
cat("Top ranked genes:\n")
print(head(gene_list))

# Save mapped ranked list
write.csv(
  res_mapped_unique %>%
    arrange(desc(stat)),
  file.path(enrich_outdir, "astro_mapped_ranked_genes_entrez.csv"),
  row.names = FALSE
)

# ============================================================
# 5. GSEA: GO Biological Process
# ============================================================

set.seed(42)

gsea_go_bp <- gseGO(
  geneList = gene_list,
  OrgDb = org.Hs.eg.db,
  ont = "BP",
  keyType = "ENTREZID",
  minGSSize = 10,
  maxGSSize = 500,
  pvalueCutoff = 1,
  pAdjustMethod = "BH",
  verbose = FALSE
)

gsea_go_bp_df <- as.data.frame(gsea_go_bp)

write.csv(
  gsea_go_bp_df,
  file.path(enrich_outdir, "astro_GSEA_GO_BP_results.csv"),
  row.names = FALSE
)

cat("\nTop GO BP GSEA results:\n")
print(head(gsea_go_bp_df, 20))

# Dotplot for GO BP GSEA
if (nrow(gsea_go_bp_df) > 0) {
  p_go_dot <- dotplot(
    gsea_go_bp,
    showCategory = 15,
    split = ".sign"
  ) +
    facet_grid(. ~ .sign) +
    ggtitle("Astrocyte GSEA: GO Biological Process")
  
  ggsave(
    file.path(enrich_outdir, "astro_GSEA_GO_BP_dotplot.png"),
    p_go_dot,
    width = 11,
    height = 6,
    dpi = 300
  )
}

# Ridgeplot for GO BP GSEA
if (nrow(gsea_go_bp_df) > 0) {
  p_go_ridge <- ridgeplot(
    gsea_go_bp,
    showCategory = 15
  ) +
    ggtitle("Astrocyte GSEA: GO Biological Process")
  
  ggsave(
    file.path(enrich_outdir, "astro_GSEA_GO_BP_ridgeplot.png"),
    p_go_ridge,
    width = 10,
    height = 7,
    dpi = 300
  )
}

# ============================================================
# 6. GSEA: MSigDB Hallmark pathways
# ============================================================

# Load Hallmark pathways
hallmark_sets <- msigdbr(
  species = "Homo sapiens",
  category = "H"
) %>%
  dplyr::select(gs_name, entrez_gene)

gsea_hallmark <- GSEA(
  geneList = gene_list,
  TERM2GENE = hallmark_sets,
  minGSSize = 10,
  maxGSSize = 500,
  pvalueCutoff = 1,
  pAdjustMethod = "BH",
  verbose = FALSE
)

gsea_hallmark_df <- as.data.frame(gsea_hallmark)

write.csv(
  gsea_hallmark_df,
  file.path(enrich_outdir, "astro_GSEA_Hallmark_results.csv"),
  row.names = FALSE
)

cat("\nTop Hallmark GSEA results:\n")
print(head(gsea_hallmark_df, 20))

if (nrow(gsea_hallmark_df) > 0) {
  p_hallmark_dot <- dotplot(
    gsea_hallmark,
    showCategory = 15,
    split = ".sign"
  ) +
    facet_grid(. ~ .sign) +
    ggtitle("Astrocyte GSEA: MSigDB Hallmark pathways")
  
  ggsave(
    file.path(enrich_outdir, "astro_GSEA_Hallmark_dotplot.png"),
    p_hallmark_dot,
    width = 11,
    height = 6,
    dpi = 300
  )
}

# ============================================================
# 7. Over-representation analysis using significant DE genes
# ============================================================

sig_genes <- res_mapped %>%
  filter(!is.na(padj), padj < 0.05, abs(log2FoldChange) >= 0.5)

sig_up <- sig_genes %>%
  filter(log2FoldChange > 0) %>%
  pull(ENTREZID) %>%
  unique()

sig_down <- sig_genes %>%
  filter(log2FoldChange < 0) %>%
  pull(ENTREZID) %>%
  unique()

background_genes <- res_mapped$ENTREZID %>%
  unique()

cat("\nSignificant genes used for ORA:\n")
cat("Upregulated in AD:", length(sig_up), "\n")
cat("Downregulated in AD:", length(sig_down), "\n")

# -----------------------------
# ORA for upregulated genes
# -----------------------------

if (length(sig_up) >= 10) {
  ego_up <- enrichGO(
    gene = sig_up,
    universe = background_genes,
    OrgDb = org.Hs.eg.db,
    keyType = "ENTREZID",
    ont = "BP",
    pAdjustMethod = "BH",
    pvalueCutoff = 0.1,
    qvalueCutoff = 0.2,
    readable = TRUE
  )
  
  ego_up_df <- as.data.frame(ego_up)
  
  write.csv(
    ego_up_df,
    file.path(enrich_outdir, "astro_ORA_GO_BP_upregulated_AD.csv"),
    row.names = FALSE
  )
  
  if (nrow(ego_up_df) > 0) {
    p_ora_up <- dotplot(
      ego_up,
      showCategory = 15
    ) +
      ggtitle("GO BP enrichment: genes upregulated in AD astrocytes")
    
    ggsave(
      file.path(enrich_outdir, "astro_ORA_GO_BP_upregulated_AD_dotplot.png"),
      p_ora_up,
      width = 9,
      height = 6,
      dpi = 300
    )
  }
} else {
  cat("Skipping ORA for upregulated genes: fewer than 10 significant genes.\n")
}

# -----------------------------
# ORA for downregulated genes
# -----------------------------

if (length(sig_down) >= 10) {
  ego_down <- enrichGO(
    gene = sig_down,
    universe = background_genes,
    OrgDb = org.Hs.eg.db,
    keyType = "ENTREZID",
    ont = "BP",
    pAdjustMethod = "BH",
    pvalueCutoff = 0.1,
    qvalueCutoff = 0.2,
    readable = TRUE
  )
  
  ego_down_df <- as.data.frame(ego_down)
  
  write.csv(
    ego_down_df,
    file.path(enrich_outdir, "astro_ORA_GO_BP_downregulated_AD.csv"),
    row.names = FALSE
  )
  
  if (nrow(ego_down_df) > 0) {
    p_ora_down <- dotplot(
      ego_down,
      showCategory = 15
    ) +
      ggtitle("GO BP enrichment: genes downregulated in AD astrocytes")
    
    ggsave(
      file.path(enrich_outdir, "astro_ORA_GO_BP_downregulated_AD_dotplot.png"),
      p_ora_down,
      width = 9,
      height = 6,
      dpi = 300
    )
  }
} else {
  cat("Skipping ORA for downregulated genes: fewer than 10 significant genes.\n")
}

# ============================================================
# 8. Focused AD-relevant pathway terms
# ============================================================

# This section helps you pull out interpretable AD-relevant pathways
# from the GO BP GSEA results.

ad_keywords <- c(
  "astrocyte",
  "glial",
  "gliosis",
  "inflammatory",
  "immune",
  "cytokine",
  "synapse",
  "synaptic",
  "neuron",
  "mitochond",
  "oxidative",
  "apoptotic",
  "protein folding",
  "response to stress",
  "lipid",
  "amyloid",
  "tau"
)

pattern <- paste(ad_keywords, collapse = "|")

focused_go_terms <- gsea_go_bp_df %>%
  filter(str_detect(tolower(Description), pattern)) %>%
  arrange(p.adjust)

write.csv(
  focused_go_terms,
  file.path(enrich_outdir, "astro_GSEA_GO_BP_AD_relevant_terms.csv"),
  row.names = FALSE
)

cat("\nFocused AD-relevant GO terms:\n")
print(head(focused_go_terms, 30))

# ============================================================
# 9. Save top summary tables
# ============================================================

top_go_summary <- gsea_go_bp_df %>%
  arrange(p.adjust) %>%
  dplyr::select(ID, Description, NES, pvalue, p.adjust, qvalue, core_enrichment) %>%
  slice_head(n = 30)

top_hallmark_summary <- gsea_hallmark_df %>%
  arrange(p.adjust) %>%
  dplyr::select(ID, Description, NES, pvalue, p.adjust, qvalue, core_enrichment) %>%
  slice_head(n = 30)

write.csv(
  top_go_summary,
  file.path(enrich_outdir, "astro_top30_GO_BP_GSEA_summary.csv"),
  row.names = FALSE
)

write.csv(
  top_hallmark_summary,
  file.path(enrich_outdir, "astro_top30_Hallmark_GSEA_summary.csv"),
  row.names = FALSE
)

# ============================================================
# 10. Final message
# ============================================================

cat("\nModule 3 Analysis 2 enrichment complete.\n")
cat("Results saved to:", enrich_outdir, "\n")