## ML-Ops Project 2

### Classical Data Analysis

- Öncelikle gerekli kütüphanaleri ekledim
- Daha sonrasında BootCamp eğiticisinin GitHub Repo'sundan ödev olarak verilen datayı çektim.
    * ```df = pd.read_csv('https://github.com/erkansirin78/datasets/raw/master/housing.csv')```
    * Aslında bu komutta bir API komutu
- Daha sonrasında keşifçi veri analizi yaptım kısaca
- 'ocean_proximity' harici değişkenler sayısal ve paireed olduğu için aralarındaki korelasyonu inceledim

- Ardından tekrar paired ilişkiler için scatterplot ile görselleştirdim.
- Burada aralarındaki ilişkilerin gücünü değerlendirmek önemli
- Daha sonra kayıp değerleri değerlendirdim. Burada zaten missing value ile handling'i PyCaret ile birlikte yapacağım.
- Daha sonrasında geographic bilgiler olması nedeniyle bunları görüntülemek istedim    
- Beyinlerimizin görsel desenleri doğal olarak tanıma yeteneği vardır, ancak bu desenleri etkili bir şekilde vurgulamak için görselleştirmeleri manipüle etmek önemlidir. Bu şekilde Amerika'nın batısına benzemekteydi.
- Kaliforniya haritasını düşündüğümüzde, Körfez Bölgesi, Los Angeles ve San Diego gibi yoğun nüfuslu bölgelerin kolaylıkla tanınabilir olduğu açık hale gelir.
- Konumlara göre görüntülemede 4. bir boyut için önce 'population' ile scatter'lerin boyutlarını belirledim ve ardından 'median_house_value' değerine göre renklendirdim.
- Peki bunu ısıl harita ile de göstermem mümkün müydü? Elbette, daha önce Kaggle'da karşılaştığım şekilde `folium` kütüphanesini kullnarak ev ilanlarının sıklığına göre ısı haritasını oluşturdum.
- `ocean_proximity`'e göre sayılarını görselletştirdim.

### Ml - Ops
- Daha sonra artık MlOps kısmına giriş yapalım;
- Week_1'de anlattığım üzere kurulumlar ile MlFlow ve MySQL'i kullanıyorum. Bu containerler anlık olarak Oracle infra structure üzerinde aktif çalışıyorlar. Sürekli bağlanabiliyorum.
- Gerekli enviromentleri kuruyoruz ben burada `localhost` kullandım ancak sizler kendi `ip adresinizi` veya `özel configürasyonlarınızı` kullanabilirsiniz
- PyCaret'in bu şekilde bir config istediğini bilmiyordum benim de esinlenmiş olduğum [Moez Ali'nin `PyCaret'in geliştiricisi`](https://moez-62905.medium.com/simplify-mlops-with-pycaret-mlflow-and-dagshub-366c768f0dac) ve bu nootebook da faydalandığım medium'u okuyabilirsiniz.

- Standart kurulmda eklememiz gereken argümanlar şu şekilde;
```
log_experiment = True,
log_data=True,
#log_profile=True, # wasting time
log_plots=True,
system_log = True,
experiment_name='MLFlow_by_PyCaret'
```
- Burada önemli olan şey şu, PyCaret `.env` dosyası varsa bunu otomatik olarak alıyor bu yüzden `.env` dosyası içerisinde gereksiz anahtarlar olursa hata alırsınız.
- `experiment_name` seçimini unique yapmalısınız değişebiliyor, veya aktif kaldığını söylüyor.

#### Lets start to experiments
- PyCaret tüm model denemelerini otomatik olarak artık mlflow'un experiment kısmına atıyor. Tüm denemeleri ise ilk başlama ile oluşturulan setup üzerinde birleştiriyor yani tek sayfa gibi gözüksede `+` kısmına basarak kaydedilen tüm deneyleri görmeniz mümkün. Yakın zamanda PyCaret güncellemesiyle bazı idle'larda html sonuçları görüntülenemiyordu. Bu şekilde benim daha bile kolayıma geldi gerçekten.
- Mesela `compare_model()` içerisinde ki tüm modellerin logları otomatik olarak aktarılıyor. sadece bu modelleri değil tüm denenen modellerin sonuçlarını karşılaştırabileceğiniz interaktif bir tablo ile çalışmak muhakkak büyük keyif.
- bastırılan tüm `plot`'lar da yine aynı şekilde loglanıyor ve dilediğiniz şekilde istediğiniz zaman bakmak isteidğiniz MySQL'inizden çekiliyor.

#### Neyse modeli bitirelim ve en iyi modelleri saklayalım;

- Artık seçtiyseniz en iyi modellerinizi kayıt edebilirsiniz, bunun için basit iteratif bir fonksiyon yarattım

- Bu kısımda bariz bir sıkıntı var ama;
- PyCaret modelleri MLFlow'a yönlendirildiklerinde otomatik olarak performans metriklerini, hiperparametreleri ve diğer detayları kaydeder. Ancak, modelin kendisini kaydetmek isterseniz farklı bir yaklaşım vardır. Bu yaklaşımda, öncelikle PyCaret modelini finalleştirir ve ardından log_model işlevini kullanarak işlemi tamamlarız. Şablon kodunda sklearn örneği olarak kullanılmış olsa da, bazı ML kütüphaneleri için (örneğin PyTorch veya FastAI) özel işlevler de mevcuttur. Gelecekte PyCaret'in böyle bir özel işlevi olup olmayacağını bilmiyorum. Henüz üstesinden gelmediğim bir zorluk, modelleri sürümler halinde kaydetmektir. Bununla birlikte, modellerin yalnızca bir sürümünü yüklemek ve her biri için ayrı isimler kullanmak mümkündür. Ancak ben kişisel olarak bir deneme sürecini modelin sürümleri olarak kaydetmeyi daha mantıklı buluyorum. Modellere girdiğinizde, zaten hangi algoritmanın kullanıldığını belirtir.
- Bu işlem tamamlandıktan sonra modellerinizi Restered Model kısmında inceleyebilirsiniz.

#### Peki silmek istedik veya browserda açmadan  `registered modeller`'i listelemek istedik;

- Bunun için de kullanılabilir bir fonksiyon tanımladım, enviroment configuration'larını otomatik olarak alıyor.

- Ml-Ops projelerimi takipte kalın lütfen.