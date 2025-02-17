rm(list = ls())
library(clusterProfiler)
library(org.Hs.eg.db)
library(enrichplot)
library(ggplot2)
library(parallel)
library(doParallel)
library(this.path)
library(optparse)
setwd(dirname(this.path()))

option_list <- list(
  make_option(c("-n", "--num_cores"), type = "integer", default = 2, help = "Number of cores to use for makeCluster", metavar = "integer"),
  make_option(c("-d", "--save_dir"), type = "character", default = "./output", help = "Directory to save the results", metavar = "character")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

num_cores <- opt$num_cores
save_dir <- opt$save_dir

load('../src/gson_GO_all.rData')
gsid2gene <- gson_GO_all@gsid2gene
gene2name <- gson_GO_all@gene2name
gsid2name <- gson_GO_all@gsid2name
gsid2ont <- go2ont(gsid2name$gsid)

hsGO_BP_IC <- read.csv('../src/hsGO_BP_IC.csv')
rownames(hsGO_BP_IC) <- hsGO_BP_IC$GO

uniprot2symbol_table <- read.csv('../src/uniprot2symbol_all.csv')
colnames(uniprot2symbol_table) <- c('uniprot', 'symbol')

#### GO ####
GO_enrichment <- function(pred.res, ont = "BP", gene_number, q_cutoff) {
  enrich_GO <- enrichGO(
    pred.res,
    org.Hs.eg.db,
    keyType = "SYMBOL",
    ont = ont,
    pvalueCutoff = q_cutoff,
    pAdjustMethod = "BH",
    minGSSize = 1,
    maxGSSize = gene_number,
    readable = FALSE,
    pool = FALSE)
  return(enrich_GO)}

df2go <- function(df, ont = "BP", gene_number = 2500, q_cutoff = .05){
  df <- df$protein
  df <- uniprot2symbol_table[uniprot2symbol_table$uniprot %in% df, 'symbol']
  df.go.list <- GO_enrichment(df, ont, gene_number, q_cutoff)
  return(df.go.list)}

plot_treeplot <- function(go.obj, showCategory = 20, color = 'IC', color_low = "red", color_high = "blue", nCluster = 5){
  go.obj@result$rank <- 1:nrow(go.obj@result)
  go.obj@result$Count <- rank(go.obj@result$p.adjust)
  go.obj@result$p.adj.log <- -log10(go.obj@result$p.adjust)
  go.obj@result$IC <- hsGO_BP_IC[go.obj@result$ID, 'IC']
  similarity <- pairwise_termsim(go.obj)
  treeplot(similarity,
      showCategory = showCategory, color = color, label_format=NULL, fontsize=4, 
      hilight.params=list(hilight=T, align="both")
      ,offset.params=list(bar_tree=rel(1), tiplab=rel(1),extend=0.3,hexpand=0.1)
      ,cluster.params=list(method="ward.D",n=nCluster,color=NULL, label_words_n=4,label_format=30)
      ) + scale_size_continuous(name = "rank", range = c(8, 3) * 1)+
      scale_color_gradient(low = color_high, high = color_low, name = color)
       }

tidy <- function(df){
  df$BgRatio <- as.numeric(gsub("/18888", "", df$BgRatio))
  df1 <- df[df$BgRatio <= 200,][1:10,]
  df2 <- df[df$BgRatio > 200,][1:10,]
  df.ult <- rbind(df1, df2)
  df.ult <- df.ult[order(df.ult$p.adjust, decreasing = F),]
  df.ult$IC <- hsGO_BP_IC[match(df.ult$ID, hsGO_BP_IC$GO), 'IC'] 
  return(df.ult)
}

cl <- makeCluster(num_cores)
clusterExport(cl, varlist = c("save_dir"))
registerDoParallel(cl)
print('Annotation start...')

pred.files <- dir('../tmp/', pattern = '.csv')
foreach(i=1:length(pred.files), .packages = c("clusterProfiler", "org.Hs.eg.db", "enrichplot", "ggplot2")) %dopar% {
    print(paste0(i,'/',length(pred.files)))
    file <- pred.files[i]
    id <- unlist(strsplit(file, '_'))[1]
    pred <- read.csv(paste0('../tmp/', file))
    pred <- pred[c(1:round(nrow(pred)/3)),]
    
    # GO enrichment
    go.bp <- df2go(pred, 'BP', gene_number = 2500, q_cutoff = 5e-2)
    go.bp.df <- as.data.frame(go.bp) 
    go.bp.df <- tidy(go.bp.df)
    write.table(go.bp.df, paste0(save_dir, id, '_GO.tsv'), row.names = F, quote = F, sep = '\t')
    
    # viz
    pdf(paste0(save_dir, id, '_go_bp.ranked.pdf'), width = 12, height = 9)
    go.bp@result <- na.omit(go.bp.df)
    treeplot <- plot_treeplot(go.bp, showCategory = nrow(na.omit(go.bp.df)), nCluster = 6)
    print(treeplot)
    dev.off()

    pdf(paste0(save_dir, id, '_go_bp.dag.pdf'), width = 12, height = 9)
    dag <- plotGOgraph(go.bp, firstSigNodes = nrow(na.omit(go.bp.df)))
    print(dag)
    dev.off()
}

stopCluster(cl)