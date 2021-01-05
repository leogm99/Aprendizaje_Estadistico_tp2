library(MASS);
library(ggplot2);
library(GGally);
library(ggrepel);
library(caret);
library(MVN);
library(tidyverse);
library(car)
library(xtable)
data("iris");
# lucky 23
set.seed(23);

# funcion que hace lda en un subset de iris
# devuelve las proyecciones

train_iris <- function(){
  model <- MASS::lda(Species ~ ., iris);
  preds <- predict(model, iris[,1:4]);
  preds.df <- data.frame(preds$x);
  preds.df["Species"] <- iris$Species;
  print(ggplot(preds.df, aes(x = LD1, y = LD2, label = Species, color = Species)) + 
    geom_point() +  geom_label_repel(aes(label = Species),
                                     box.padding   = 0.2, 
                                     point.padding = 0.3,
                                     segment.color = 'grey50',
                                     max.overlaps = 10) +
      ggtitle("Proyección sobre los ejes principales") +
    theme(plot.title = element_text(hjust = 0.5), legend.position = "none"));
  conf_matrix <- confusionMatrix(data = preds$class, reference = iris$Species)
  return(conf_matrix);
}


# lda con cv
train_iris_cv <- function(sample_size, train_size) {
  train_subset <- sample(1:sample_size, train_size);

  z <- MASS::lda(Species ~ ., iris, subset = train_subset);
  species_test <- select(iris, Species)[-train_subset, ];
  preds <- predict(z, iris[-train_subset, ]);
  preds_train <- predict(z, iris[train_subset, ]);
  print(confusionMatrix(preds_train$class, iris[train_subset, ]$Species));
  preds.df <- data.frame(preds$x);
  preds.df["Species"] <- species_test;
  print(ggplot(preds.df, aes(x = LD1, y = LD2, label = Species, color = Species)) + 
          geom_point() +  geom_label_repel(aes(label = Species),
                                           box.padding   = 0.2, 
                                           point.padding = 0.3,
                                           segment.color = 'grey50',
                                           max.overlaps = 10) +
          ggtitle("Proyección sobre los ejes principales - Test set") +
          theme(plot.title = element_text(hjust = 0.5), legend.position = "none"));
  conf_matrix <- confusionMatrix(
                     preds$class, iris[-train_subset, ]$Species);
  return(conf_matrix);
}


# heatmap para las especies segun su petal y sepal width

split_species <- function(){
  return(iris %>% split(iris$Species));
}

heatmap <- function(iris_data, title){
  return(ggplot(data = iris_data, aes(x=Sepal.Width, y=Petal.Width) ) +
        stat_density_2d(aes(fill = ..density..), geom = "raster", contour = FALSE) +
        scale_fill_distiller(palette = "Spectral", direction = 1) +
        scale_x_continuous(expand = c(0, 0)) +
        scale_y_continuous(expand = c(0, 0)) +
        theme(
          legend.position='none'
        ) + ggtitle(paste(title)));
}

build_heatmaps <- function(){
  species <- split_species() %>% map(data.frame) %>% map(select, Sepal.Width, Petal.Width);
  names <- c("Setosa Density", "Versicolor Density", "Virginica Density");
  Map(heatmap, species, names);
}

build_contours <- function(){
  species <- split_species() %>% map(data.frame) %>% map(select, Sepal.Width, Petal.Width);
  Map(MVN::mvn, species, multivariatePlot = "contour");
}

k_fold <- function(train_method){
  best_k <- 0;
  best_acc <- 0;
  actual_acc <- 0;
  model_res <- 0;
  best_model <- 0;
  for (k in 2:150) {
    
    # cross-validation, k = k
    train_control <- trainControl(method = "cv", number = k);
    
    # el modelo es un lda (3 especies, proyecta a R2)
    
    model <- train(Species ~.,
                   data = iris, method = train_method,
                   trControl = train_control);
    actual_acc <- model$results$Accuracy;
    if(actual_acc > best_acc){
      best_acc <- actual_acc;
      model_res <- model$results;
      best_k <- k;
      best_model <- model;
    }
  }
  return(list(best_k, model_res, best_model));
}

iris_sepal_petal_scatter <- function(iris_data, title){
  title <- title;
  color <- 'darkblue';
  if("Species" %in% colnames(iris_data))
    color <- iris_data$Species;
  ggplot(iris_data) + geom_point(aes(x = Sepal.Width, y = Petal.Width), color = color) +
  ggtitle(title);
}

build_scatters <- function(){
  print(ggplot(iris) + geom_point(aes(x = Sepal.Width, y = Petal.Width, color = Species)) + 
          ggtitle("Sépalo vs Pétalo | Ancho por especie"));
  sepal_petal <- split_species() %>% map(~ select(.x, Sepal.Width, Petal.Width));
  names <- c("Ancho de Sépalo Vs Pétalo | Setosa", "Ancho de Sépalo Vs Pétalo | Versicolor", "Ancho de Sépalo Vs Pétalo | Virginica")
  Map(iris_sepal_petal_scatter, sepal_petal, names);
}

#helpers, devuelven el width para los petalos y sepalos de cada especie

make_petals_width <- function(){
  # reparar esto
  return(split_species() %>% map(~ select(.x, Petal.Width)));
} 

make_sepals_width <- function(){
  return(split_species() %>% map(~ select(.x, Sepal.Width)));
} 

iris_1d_density <- function(iris_data, title){
  ggplot(iris_data, aes(x = iris_data[,1])) + geom_density() + ggtitle(title) + xlab("Ancho");
}

build_iris_1d_densities <- function(){
  petal_names <- c("Densidad del ancho de pétalo | Setosa",
                   "Densidad del ancho de pétalo | Versicolor",
                   "Densidad del ancho de pétalo | Virginica");
  sepal_names <- c("Densidad del ancho de sépalo | Setosa",
                   "Densidad del ancho de sépalo | Versicolor",
                   "Densidad del ancho de sépalo | Virginica");
  
  petals <- make_petals_width();
  sepals <- make_sepals_width();
  print(Map(iris_1d_density, petals, petal_names));
  print(Map(iris_1d_density, sepals, sepal_names));
}

## devuelve el p-valor del test de shapiro-wilk para los petalos en las 3 especies
## [setosa, versicolor, virginica], se testea con nivel de significacion 0.05
## h0: es normal vs h1: se distribuye de otra manera
# si rechazo h0 se que no se distribuye de manera normal 

test_normality_petals <- function(){
  return(make_petals_width() %>% map(~unlist(.x)) %>% map(~shapiro.test(.x)))
}

test_normality_sepals <- function(){
  return(make_sepals_width() %>% map(~unlist(.x)) %>% map(~shapiro.test(.x)));
}

## funcion que tome prestada de 
# https://stackoverflow.com/questions/63782598/quadratic-discriminant-analysis-qda-plot-in-r

decisionplot_ggplot <- function(model, data, class = NULL, predict_type = "class",
                                resolution = 100, showgrid = TRUE, ...) {
  
  if(!is.null(class)) cl <- data[,class] else cl <- 1
  data <- data[,1:2]
  cn <- colnames(data)
  
  k <- length(unique(cl))
  
  data$pch <- data$col <- as.integer(cl) + 1L
  gg <- ggplot(aes_string(cn[1], cn[2]), data = data) +  geom_point() + 
    geom_point(aes(x = 3.5, y = 1.75), colour="black") + 
    geom_point(aes_string(col = 'as.factor(col)', shape = 'as.factor(col)'), size = 3)
  
  # make grid
  r <- sapply(data[, 1:2], range, na.rm = TRUE)
  xs <- seq(r[1, 1], r[2, 1], length.out = resolution)
  ys <- seq(r[1, 2], r[2, 2], length.out = resolution)
  
  g <- cbind(rep(xs, each = resolution), 
             rep(ys, time = resolution))
  colnames(g) <- colnames(r)
  
  g <- as.data.frame(g)
  
  ### guess how to get class labels from predict
  ### (unfortunately not very consistent between models)
  p <- predict(model, g, type = predict_type)
  if(is.list(p)) p <- p$class
  g$col <- g$pch <- as.integer(as.factor(p)) + 1L
  
  if(showgrid) 
    gg <- gg + geom_point(aes_string(x = cn[1], y = cn[2], col = 'as.factor(col)'), data = g, shape = 20, size = 1)
  
  gg + geom_contour(aes_string(x = cn[1], y = cn[2], z = 'col'), data = g, inherit.aes = FALSE)
}

lda_iris <- train_iris();
print(lda_iris);
xtable(lda_iris$table);

# 70% del set original para entrenar
# es un parametro a tunear (fiaquita)
sample_size <- length(iris$Species);
train_size <- 105;
train_subset <- sample(1:sample_size, train_size);
conf_matrix_cv <- train_iris_cv(sample_size, train_size);
print(conf_matrix_cv)

xtable(conf_matrix_cv$table)


#qda 

qda_iris <- qda(Species ~ ., iris)
qda_pred <- predict(qda_iris, iris)
xtable(confusionMatrix(reference = iris$Species, data = qda_pred$class)$table)

sample_size <- length(iris$Species);
train_size <- 105;
train_subset <- sample(1:sample_size, train_size);
qda_cv <- qda(Species ~ ., iris[train_subset, ]);
preds_qda_cv <- predict(qda_cv, iris[-train_subset, ]);
xtable(confusionMatrix(reference = iris[-train_subset, ]$Species, data = preds_qda_cv$class)$table)
preds_train_qda_cv <- predict(qda_cv, iris[train_subset, ]);
xtable(confusionMatrix(reference = iris[train_subset, ]$Species, data = preds_train_qda_cv$class)$table)



# pair plot

ggpairs(iris, upper = list(continuous = wrap("cor", family = "sans")));

#boxplot y density interesantes
ggplot(data = iris, aes(Petal.Length, fill="red")) + 
  geom_density(alpha = 0.1) +
  theme(legend.position = "none") + 
  xlab("Largo del pétalo") +
  ylab("Densidad")

ggplot(iris, aes(x=Species, y=Petal.Length, fill=Species)) + 
  geom_violin(trim=FALSE)+
  geom_boxplot(width=0.1, fill="white")+
  labs(title="Largo de los pétalos según especie", x="Especie", y = "Largo") + 
  theme(plot.title = element_text(hjust = 0.5))


# a) Suponiendo normalidad en los ,datos, hacer una clasificacion lineal
#    y calcular el error de clasificacion de dos maneras:
#    1) proporcion de datos mal clasificados (error aparente)
#    2) cross validation (k a eleccion?)

conf_matrix <- train_iris_cv(sample_size, train_size);
accuracy <- conf_matrix$overall["Accuracy"]

# cross validation
# que k usamos ? todos los posibles
# usamos caret

k_fold_lda <- k_fold("lda");
print(k_fold_lda[[2]])

k_fold_qda <- k_fold("qda");
# clasificacion cuadratica

# 2 
# graficar los pares (x2, x4) en el plano
# ver si provienen de una distribucion normal bivariada

build_scatters();

build_heatmaps();

build_contours();

# parece ser que versicolor y virginica podrian provenir de una normal bivariada 
# se que si la conjunta es normal, entonces las marginales son normales
build_iris_1d_densities()
test_normality_petals()
shapiro_tests <- test_normality_sepals()
# claramente la setosa no podria ser jamas normal

# b) Suponiendo normal bivariada (ancho sepalo / petalo) 
# construir la regla de clasificacion cuadratica, asumiendo p = 1/3 para cada
# poblacion. Clasificar la observacion x0 = (3.5, 1.75) como perteneciente
# a alguno de los 3 grupos
# ademas tiene el plot de decision boundary<

datos <- select(iris, Sepal.Width, Petal.Width, Species)
width_model <- qda(Species ~ ., datos, prior = c(1/3, 1/3, 1/3))
preds <- predict(width_model, datos)
x0 <- data.frame("Sepal.Width" = 3.5, "Petal.Width" = 1.75);
sum(preds$class == datos$Species) / 150
# prediccion de (3.5, 1.75) -> versicolor!
xtable(predict(width_model, x0)$posterior)
xtable(confusionMatrix(data = preds$class, reference = iris$Species)$table)

decisionplot_ggplot(width_model, datos, class = "Species") 

# c) supongamos que las matrices de covarianza son iguales para las 3 poblaciones
# normales bivariadas -> de qda pasamos a lda. clasificar x0 = (3.5, 1.75)
# plotear la decision boundary
# comparar los resultados con qda

width_lda_model <- lda(Species ~ ., datos, prior = c(1/3, 1/3, 1/3))
preds_lda <- predict(width_lda_model, datos)
sum(preds_lda$class == datos$Species) / 150
xtable(predict(width_lda_model, x0)$posterior)
decisionplot_ggplot(width_lda_model, datos, class = "Species")
preds_lda$class
c <- confusionMatrix(data = preds_lda$class, reference = iris$Species)
xtable(c$table)
# comparamos las matrices de covarianza 

cov_per_species <- split_species() %>% map(~ select(.x, Sepal.Width, Petal.Width)) %>% map(~ cov(.x))
cov(datos[, -3])

xtable(cov_per_species$setosa)
xtable(cov_per_species$versicolor)
xtable(cov_per_species$virginica)


table_sample_size <- 15;
table_size <- 150
table_sample <- sample(1:150, size = 15);
xtable(iris[table_sample, ])
