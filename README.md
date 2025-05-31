# goruntusiniflamagradio
goruntu siniflama gradio
Hedefimiz Ne?
Bu kodun amacı, bir resim yüklediğimizde o resimde ne olduğunu tahmin eden (örneğin "bu bir kedi" veya "bu bir araba") bir web sayfası oluşturmak. Bunu yaparken de çok güçlü bir yapay zeka modeli kullanacağız ve web sayfasını kolayca oluşturmak için Gradio adlı bir araçtan faydalanacağız.
Kullanacağımız Ana Araçlar (Kütüphaneler):
PyTorch (torch, torchvision):
Ne İşe Yarar? PyTorch, yapay zeka ve özellikle derin öğrenme modelleri oluşturmak ve eğitmek için kullanılan çok popüler bir kütüphane (yani hazır kod koleksiyonu). "Tensor" adı verilen özel bir veri yapısıyla çalışır (sayıların çok boyutlu dizileri gibi düşünebilirsiniz).
torchvision ise PyTorch'un görüntü işleme görevleri için özel olarak geliştirilmiş bir parçası. Hazır modeller, veri setleri ve görüntü dönüştürme araçları içerir.
Pillow (PIL) (from PIL import Image):
Ne İşe Yarar? Pillow, Python'da resim dosyalarını açmak, değiştirmek, kaydetmek gibi işlemler yapmak için kullanılan bir kütüphane. Resimleri programımızın anlayabileceği bir formata getirir.
Requests (import requests):
Ne İşe Yarar? Requests, internetten veri indirmek (örneğin bir web sayfasından dosya çekmek) için kullanılan basit ve kullanışlı bir kütüphane.
Gradio (import gradio as gr):
Ne İşe Yarar? Gradio, yapay zeka modellerimiz için çok hızlı ve kolay bir şekilde web arayüzleri oluşturmamızı sağlayan bir kütüphane. Kod yazma yükünü azaltır.
Şimdi adımlara geçelim:
Adım 1: Görüntü Sınıflandırma Modelini Hazırlamak
Bilgisayarımızın bir resimde ne olduğunu anlaması için "eğitilmiş" bir modele ihtiyacı var. Bu model, daha önce milyonlarca resim görmüş ve hangi resimde ne olduğunu öğrenmiş akıllı bir program gibi.
import torch
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).eval()
Use code with caution.
Python
import torch:
Anlamı: "Ey Python, birazdan PyTorch kütüphanesinin fonksiyonlarını kullanacağım, onu çağır ve hazırla." diyoruz. Artık torch kelimesini kullanarak PyTorch'un özelliklerine erişebiliriz.
Girdi Türü: Yok (sadece kütüphaneyi yüklüyor).
Çıktı Türü: Yok (sadece kütüphaneyi yüklüyor).
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True):
Anlamı: Bu satır, PyTorch Hub adı verilen bir yerden hazır, eğitilmiş bir görüntü sınıflandırma modeli indiriyor.
torch.hub.load(...): PyTorch Hub'dan bir şeyler yüklemek için kullanılan komut.
'pytorch/vision:v0.6.0': Modelin bulunduğu yer (PyTorch'un görüntü kütüphanesinin belirli bir versiyonu).
'resnet18': İndirmek istediğimiz modelin adı. ResNet-18, oldukça iyi performans gösteren ve çok büyük olmayan bir model türü.
pretrained=True: "Bu modeli daha önce başkaları tarafından eğitilmiş haliyle istiyorum, sıfırdan eğitmekle uğraşmayacağım." anlamına gelir. Bu sayede model zaten birçok şeyi tanıyabiliyor olacak.
Girdi Türü: Modelin adı ve ayarları (metin olarak).
Çıktı Türü: İndirilmiş ve yüklenmiş yapay zeka modeli. Bu modeli model adında bir değişkene atıyoruz. Artık model dediğimizde bu akıllı programı kastediyor olacağız.
.eval():
Anlamı: Modelimizi indirdik, şimdi ona "Değerlendirme moduna geç, yani yeni resimleri tahmin etmeye hazır ol. Eğitim modunda değilsin." diyoruz. Bu, modelin bazı iç ayarlarını tahmin yapmaya uygun hale getirir.
Girdi Türü: Yok (önceki satırdan gelen model üzerinde çalışır).
Çıktı Türü: Değerlendirme moduna ayarlanmış model (aslında aynı model değişkenini günceller).
Bu adımın sonunda ne oldu? Artık model adlı bir değişkenimiz var ve bu değişken, resimleri sınıflandırmaya hazır, önceden eğitilmiş bir ResNet-18 yapay zeka modelini tutuyor.
Adım 2: Tahmin Fonksiyonunu Tanımlamak
Şimdi, kullanıcı bir resim yüklediğinde bu resmi alıp modelimize soracak ve modelin cevabını (yani tahminlerini) bize anlamlı bir şekilde verecek bir fonksiyon (yani bir grup komut) yazmamız gerekiyor.
import requests
from PIL import Image
from torchvision import transforms

# ImageNet için insanların okuyabileceği etiketleri indiriyoruz.
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

def predict(inp):
 inp = transforms.ToTensor()(inp).unsqueeze(0)
 with torch.no_grad():
  prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
 confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
 return confidences
Use code with caution.
Python
import requests:
Anlamı: İnternetten veri indirmek için requests kütüphanesini çağırıyoruz.
from PIL import Image:
Anlamı: Resim dosyalarıyla çalışmak için Pillow kütüphanesinden Image modülünü çağırıyoruz.
from torchvision import transforms:
Anlamı: PyTorch'un görüntü kütüphanesinden transforms modülünü çağırıyoruz. Bu modül, resimleri modelimizin anlayacağı formata dönüştürmek için araçlar içerir.
response = requests.get("https://git.io/JJkYN"):
Anlamı: requests kütüphanesini kullanarak verilen internet adresinden ("https://git.io/JJkYN") bir dosya indiriyoruz. Bu dosya, modelimizin tanıyabildiği 1000 farklı nesne kategorisinin insan tarafından okunabilir isimlerini (etiketlerini, örneğin "kedi", "köpek", "araba") içeriyor.
Girdi Türü: İnternet adresi (metin).
Çıktı Türü: İnternetten gelen yanıt. Bu yanıtı response değişkenine atıyoruz.
labels = response.text.split("\n"):
Anlamı: İndirdiğimiz yanıtın (response) metin içeriğini alıyoruz (.text). Bu metin, her satırda bir etiket olacak şekilde düzenlenmiş. .split("\n") komutuyla bu metni satır satır ayırıp her bir etiketi ayrı bir eleman olarak içeren bir liste oluşturuyoruz. Bu listeyi labels değişkenine atıyoruz.
Girdi Türü: response değişkeninden gelen metin.
Çıktı Türü: Etiketlerin listesi (örneğin ['tench', 'goldfish', ..., 'toilet tissue']).
def predict(inp)::
Anlamı: predict adında yeni bir fonksiyon tanımlıyoruz. Bu fonksiyon, bir parametre alacak: inp. Bu inp, kullanıcının yükleyeceği resim olacak. Fonksiyonun içindeki komutlar, bu resmi alıp işleyecek ve tahminleri döndürecek.
inp = transforms.ToTensor()(inp).unsqueeze(0):
Anlamı: Bu satır, gelen resmi modelimizin işleyebileceği özel bir formata (PyTorch tensorüne) dönüştürüyor.
transforms.ToTensor(): Bu, Pillow formatındaki bir resmi (yani inp'yi) PyTorch tensorüne çeviren bir dönüştürücü oluşturur.
(inp): Dönüştürücüyü inp resmine uygular.
.unsqueeze(0): Modelimiz genellikle bir grup resmi (bir "batch") aynı anda işlemek üzere tasarlanmıştır. Biz tek bir resim versek bile, onu sanki tek resimlik bir grupmuş gibi göstermemiz gerekir. Bu komut, tensorün başına fazladan bir boyut ekleyerek bunu yapar.
Girdi Türü: inp (Pillow formatında bir resim).
Çıktı Türü: İşlenmeye hazır PyTorch tensorü (sayısal bir temsil). Bu tensor yine inp değişkenine atanarak güncellenir.
with torch.no_grad()::
Anlamı: Bu bir "context manager". "Şimdi yapacağım işlemler sadece tahmin amaçlı, modelin öğrenmesiyle (gradyan hesaplamalarıyla) ilgili değil. Bu yüzden gereksiz hesaplamalar yapma, daha hızlı ol." anlamına gelir. Bu, tahmin yaparken bellek kullanımını azaltır ve hızı artırır.
prediction = torch.nn.functional.softmax(model(inp)[0], dim=0):
Anlamı: Bu satır asıl tahmin işlemini yapar ve sonuçları olasılıklara dönüştürür.
model(inp): Hazırladığımız inp tensorünü (resmimizi) daha önce yüklediğimiz modele veririz. Model resmi işler ve her bir kategori için bir "skor" (logit) üretir. Bu skorlar henüz olasılık değildir.
[0]: model(inp) sonucu genellikle bir "batch" için sonuçlar içerir. Biz tek resim verdiğimiz için bu "batch"in ilk (ve tek) elemanının sonuçlarını alırız.
torch.nn.functional.softmax(..., dim=0): Softmax fonksiyonu, modelden gelen ham skorları alır ve her bir kategori için 0 ile 1 arasında bir olasılığa dönüştürür. Bu olasılıkların toplamı 1 olur. dim=0 hangi boyut üzerinden olasılıkların hesaplanacağını belirtir.
Girdi Türü: inp tensorü (modele), modelden gelen skorlar (softmax'a).
Çıktı Türü: Her bir kategori için tahmin olasılıklarını içeren bir tensor. Bu, prediction değişkenine atanır.
confidences = {labels[i]: float(prediction[i]) for i in range(1000)}:
Anlamı: Bu satır, Python'da "dictionary comprehension" adı verilen şık bir yöntemle, okunabilir bir sonuç sözlüğü oluşturur.
for i in range(1000): 0'dan 999'a kadar olan sayılar için (çünkü 1000 kategorimiz var) bir döngü başlatır.
labels[i]: Döngünün her adımında, i-inci etiketi (örneğin "kedi") alır.
float(prediction[i]): Döngünün her adımında, i-inci kategori için hesaplanan olasılığı alır ve bunu ondalık sayıya (float) çevirir.
{... : ...}: Bu yapı bir sözlük oluşturur. Anahtar (key) olarak etiket adını (labels[i]), değer (value) olarak da o etikete ait güven olasılığını (float(prediction[i])) kullanır.
Girdi Türü: labels listesi ve prediction tensorü.
Çıktı Türü: Şuna benzer bir sözlük: {'kedi': 0.85, 'köpek': 0.10, ...}. Bu, confidences değişkenine atanır.
return confidences:
Anlamı: predict fonksiyonu çağrıldığında, en son hesapladığı confidences sözlüğünü sonuç olarak geri döndürür.
Bu adımın sonunda ne oldu? Artık predict adında bir fonksiyonumuz var. Bu fonksiyona bir resim verdiğimizde, o resmin her bir olası kategori için ne kadar "güvenilir" olduğunu gösteren bir sözlük (örneğin, %85 kedi, %10 köpek gibi) alacağız.
Adım 3: Gradio Arayüzünü Oluşturmak
Şimdi predict fonksiyonumuzu kullanarak kullanıcıların resim yükleyip tahminleri görebileceği bir web arayüzü oluşturacağız.
import gradio as gr

gr.Interface(fn=predict,
       inputs=gr.Image(type="pil"),
       outputs=gr.Label(num_top_classes=3),
       examples=["/content/lion.jpg", "/content/cheetah.jpg"]).launch()
Use code with caution.
Python
import gradio as gr:
Anlamı: "Ey Python, Gradio kütüphanesini çağır ve ona kısaca gr diyelim."
Girdi Türü: Yok.
Çıktı Türü: Yok.
gr.Interface(...):
Anlamı: Bu, Gradio'nun ana komutudur ve bir web arayüzü oluşturur. İçine birkaç önemli bilgi vermemiz gerekir:
fn=predict:
Anlamı: "Bu arayüzün arkasında çalışacak ana fonksiyon, daha önce tanımladığımız predict fonksiyonudur." Kullanıcı bir girdi verdiğinde Gradio bu fonksiyonu çağıracak.
Girdi Türü: Fonksiyonun adı (predict).
inputs=gr.Image(type="pil"):
Anlamı: "Kullanıcının veri gireceği alan bir resim yükleme alanı olacak."
gr.Image(...): Gradio'nun resim yükleme bileşenini oluşturur.
type="pil": "Kullanıcı bir resim yüklediğinde, o resmi Pillow (PIL) formatında predict fonksiyonumuza gönder." anlamına gelir. Hatırlarsanız predict fonksiyonumuz Pillow formatında resim bekliyordu.
Girdi Türü: Gradio giriş bileşeni tanımı.
outputs=gr.Label(num_top_classes=3):
Anlamı: "Tahmin sonuçlarını göstereceğimiz alan bir etiket (label) alanı olacak."
gr.Label(...): Gradio'nun etiket gösterme bileşenini oluşturur. Bu bileşen, predict fonksiyonumuzdan dönen sözlüğü (kategori: olasılık) alıp güzel bir şekilde gösterir.
num_top_classes=3: "Tüm 1000 kategorinin olasılığını gösterme, sadece en yüksek olasılığa sahip ilk 3 kategoriyi göster." anlamına gelir. Bu, arayüzü daha okunabilir yapar.
Girdi Türü: Gradio çıkış bileşeni tanımı.
examples=["/content/lion.jpg", "/content/cheetah.jpg"]:
Anlamı: "Arayüzde kullanıcıların tıklayıp deneyebileceği hazır örnek resimler olsun." Bu, kullanıcıların kendi resimlerini yüklemeden önce arayüzün nasıl çalıştığını görmelerini sağlar.
ÖNEMLİ NOT: Buradaki "/content/lion.jpg" ve "/content/cheetah.jpg" yolları sadece birer örnektir. Bu kodun çalıştığı bilgisayarda gerçekten bu yollarda resimler olması gerekir. Eğer yoksa, bu örnekler çalışmaz. Kendi bilgisayarınızdaki resimlerin yollarını vermelisiniz.
Girdi Türü: Resim dosyalarının yollarını içeren bir liste.
Çıktı Türü: Hazır bir Gradio arayüz nesnesi.
.launch():
Anlamı: Oluşturduğumuz Gradio arayüzünü çalıştırır ve yerel bir web sunucusu başlatır. Terminalde veya Colab gibi bir ortamda çalıştırıyorsanız, size tıklanabilir bir link verir. Bu linke tıkladığınızda web tarayıcınızda oluşturduğumuz resim sınıflandırma arayüzü açılır.
Girdi Türü: Yok (önceki satırdan gelen arayüz nesnesi üzerinde çalışır).
Çıktı Türü: Çalışan bir web uygulaması (ve genellikle terminalde bir URL).
Bu adımın sonunda ne oldu? Artık çalışan bir web sayfamız var! Bu sayfaya bir resim sürükleyip bırakabilir veya tıklayıp seçebiliriz. "Submit" (Gönder) butonuna tıkladığımızda, predict fonksiyonumuz çalışacak, resmi modelimize soracak ve en yüksek olasılıklı ilk 3 tahmini bize gösterecek. Ayrıca, verdiğimiz örnek resimlere tıklayarak da sistemi test edebiliriz.
Özetle Tüm Akış:
Model Yükleme: PyTorch Hub'dan hazır eğitilmiş bir ResNet-18 modeli indiriyoruz.
Etiketleri Alma: Modelin tanıdığı nesnelerin isimlerini (etiketlerini) internetten çekiyoruz.
predict Fonksiyonu:
Kullanıcıdan gelen resmi alır.
Resmi PyTorch tensorüne dönüştürür.
Tensorü modele verir, model skorlar üretir.
Skorları softmax ile olasılıklara çevirir.
Olasılıkları etiketlerle eşleştirip bir sözlük (kategori: güven) oluşturur.
Bu sözlüğü döndürür.
Gradio Arayüzü:
predict fonksiyonunu kullanır.
Kullanıcıdan resim girişi alır (gr.Image).
Sonuçları etiket olarak gösterir (gr.Label, en iyi 3).
Örnek resimler sunar.
Tüm bunları çalıştırıp bir web uygulaması haline getirir (.launch()).
