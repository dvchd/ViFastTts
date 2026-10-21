# Kịch bản thu bổ sung v1 (phủ âm tiết hiếm)

- Sinh từ `data/codelung/v32/train_clean_v2_dur_seg_slot4.jsonl`: **2552 type**, **600 âm tiết hiếm** (< 10 lần) được chọn.
- **1400 câu phủ** (2 câu/âm tiết, khuôn khác nhau) + **50 câu cầu nối** + **56 câu cặp tối thiểu**.
- Ước lượng: **15649 âm tiết** ≈ **58 phút** đọc (4,5 âm tiết/giây), đọc **1 lần**.
- Sau kịch bản: **600/600** âm tiết hiếm đạt ≥ 3 lần.

## Cách thu

- Cùng micro, cùng khoảng cách như bộ dữ liệu cũ (khoảng 20 cm, chếch 30°).
- **TẮT RNNoise / noise suppression / echo cancellation** trong phần mềm thu.
- Ghi **WAV gốc 48 kHz, 24-bit** (không qua Zoom/Meet/điện thoại).
- Đọc kiểu **đọc đều** (không diễn cảm), nghỉ ngắn giữa các từ hiếm.
- **Mỗi câu đọc 1 lần**; nếu vấp thì đọc lại cả câu, đừng cắt giữa câu.
- Giữ im lặng 0,5 s ở đầu/cuối mỗi file; đặt tên `v1_0001.wav` … theo đúng số câu.

## Xử lý sau khi thu

- Giữ **WAV gốc**, đưa vào pipeline `audio_v2` với cùng bước chuẩn hoá (loudness, high-pass, gate).
- 50 câu cầu nối dùng để đo chênh kênh thu (micro/phòng/mức to) giữa corpus cũ và bản mới.
- **RNNoise**: chưa chốt — sau khi có câu cầu nối, so phổ bản thô vs bản qua RNNoise offline
  với corpus cũ, chọn bản khớp hơn.

## Câu phủ âm tiết hiếm

0001. Hôm nay tôi muốn đọc từ tở thật rõ ràng.
0002. Từ tở nghe hơi lạ, nhưng rất dễ đọc.
0003. Trong câu này có từ xả, đọc chậm thôi.
0004. Tôi sẽ đọc từ xả ba lần cho bạn nghe.
0005. Tôi đã viết từ ghim lên bảng để bạn xem.
0006. Khi gặp từ ghim, bạn đừng ngắt giữa chừng.
0007. Chúng ta gặp từ lẩy trong bài đọc hôm qua.
0008. Tôi đánh dấu từ lẩy bằng bút đỏ.
0009. Mọi người cùng nhìn vào từ quân trên màn hình.
0010. Cuối buổi, tôi sẽ kiểm tra lại từ quân.
0011. Cái từ neo này xuất hiện khá ít trong sách.
0012. Bạn thử nghe lại từ neo một lần nữa nhé.
0013. Bài học hôm nay có từ chước ở ngay đầu câu.
0014. Từ chước nằm ở giữa câu, không đọc quá nhanh.
0015. Bạn có thể đọc từ nghiêng to hơn một chút.
0016. Nếu đọc sai từ nghiêng thì đọc lại từ đầu.
0017. Hôm nay tôi muốn đọc từ râu thật rõ ràng.
0018. Từ râu nghe hơi lạ, nhưng rất dễ đọc.
0019. Trong câu này có từ seo, đọc chậm thôi.
0020. Tôi sẽ đọc từ seo ba lần cho bạn nghe.
0021. Tôi đã viết từ gút lên bảng để bạn xem.
0022. Khi gặp từ gút, bạn đừng ngắt giữa chừng.
0023. Chúng ta gặp từ phác trong bài đọc hôm qua.
0024. Tôi đánh dấu từ phác bằng bút đỏ.
0025. Mọi người cùng nhìn vào từ mư trên màn hình.
0026. Cuối buổi, tôi sẽ kiểm tra lại từ mư.
0027. Cái từ chướng này xuất hiện khá ít trong sách.
0028. Bạn thử nghe lại từ chướng một lần nữa nhé.
0029. Bài học hôm nay có từ thừ ở ngay đầu câu.
0030. Từ thừ nằm ở giữa câu, không đọc quá nhanh.
0031. Bạn có thể đọc từ chử to hơn một chút.
0032. Nếu đọc sai từ chử thì đọc lại từ đầu.
0033. Hôm nay tôi muốn đọc từ hán thật rõ ràng.
0034. Từ hán nghe hơi lạ, nhưng rất dễ đọc.
0035. Trong câu này có từ hận, đọc chậm thôi.
0036. Tôi sẽ đọc từ hận ba lần cho bạn nghe.
0037. Tôi đã viết từ gặ lên bảng để bạn xem.
0038. Khi gặp từ gặ, bạn đừng ngắt giữa chừng.
0039. Chúng ta gặp từ vặn trong bài đọc hôm qua.
0040. Tôi đánh dấu từ vặn bằng bút đỏ.
0041. Mọi người cùng nhìn vào từ cò trên màn hình.
0042. Cuối buổi, tôi sẽ kiểm tra lại từ cò.
0043. Cái từ ngâm này xuất hiện khá ít trong sách.
0044. Bạn thử nghe lại từ ngâm một lần nữa nhé.
0045. Bài học hôm nay có từ xổ ở ngay đầu câu.
0046. Từ xổ nằm ở giữa câu, không đọc quá nhanh.
0047. Bạn có thể đọc từ xùa to hơn một chút.
0048. Nếu đọc sai từ xùa thì đọc lại từ đầu.
0049. Hôm nay tôi muốn đọc từ sườn thật rõ ràng.
0050. Từ sườn nghe hơi lạ, nhưng rất dễ đọc.
0051. Trong câu này có từ tẩy, đọc chậm thôi.
0052. Tôi sẽ đọc từ tẩy ba lần cho bạn nghe.
0053. Tôi đã viết từ pên lên bảng để bạn xem.
0054. Khi gặp từ pên, bạn đừng ngắt giữa chừng.
0055. Chúng ta gặp từ điêu trong bài đọc hôm qua.
0056. Tôi đánh dấu từ điêu bằng bút đỏ.
0057. Mọi người cùng nhìn vào từ cở trên màn hình.
0058. Cuối buổi, tôi sẽ kiểm tra lại từ cở.
0059. Cái từ địp này xuất hiện khá ít trong sách.
0060. Bạn thử nghe lại từ địp một lần nữa nhé.
0061. Bài học hôm nay có từ hòn ở ngay đầu câu.
0062. Từ hòn nằm ở giữa câu, không đọc quá nhanh.
0063. Bạn có thể đọc từ hô to hơn một chút.
0064. Nếu đọc sai từ hô thì đọc lại từ đầu.
0065. Hôm nay tôi muốn đọc từ khỉ thật rõ ràng.
0066. Từ khỉ nghe hơi lạ, nhưng rất dễ đọc.
0067. Trong câu này có từ trú, đọc chậm thôi.
0068. Tôi sẽ đọc từ trú ba lần cho bạn nghe.
0069. Tôi đã viết từ chập lên bảng để bạn xem.
0070. Khi gặp từ chập, bạn đừng ngắt giữa chừng.
0071. Chúng ta gặp từ trờn trong bài đọc hôm qua.
0072. Tôi đánh dấu từ trờn bằng bút đỏ.
0073. Mọi người cùng nhìn vào từ ýnh trên màn hình.
0074. Cuối buổi, tôi sẽ kiểm tra lại từ ýnh.
0075. Cái từ peo này xuất hiện khá ít trong sách.
0076. Bạn thử nghe lại từ peo một lần nữa nhé.
0077. Bài học hôm nay có từ đền ở ngay đầu câu.
0078. Từ đền nằm ở giữa câu, không đọc quá nhanh.
0079. Bạn có thể đọc từ lậu to hơn một chút.
0080. Nếu đọc sai từ lậu thì đọc lại từ đầu.
0081. Hôm nay tôi muốn đọc từ rèn thật rõ ràng.
0082. Từ rèn nghe hơi lạ, nhưng rất dễ đọc.
0083. Trong câu này có từ bướng, đọc chậm thôi.
0084. Tôi sẽ đọc từ bướng ba lần cho bạn nghe.
0085. Tôi đã viết từ gậy lên bảng để bạn xem.
0086. Khi gặp từ gậy, bạn đừng ngắt giữa chừng.
0087. Chúng ta gặp từ reo trong bài đọc hôm qua.
0088. Tôi đánh dấu từ reo bằng bút đỏ.
0089. Mọi người cùng nhìn vào từ chỉa trên màn hình.
0090. Cuối buổi, tôi sẽ kiểm tra lại từ chỉa.
0091. Cái từ bơn này xuất hiện khá ít trong sách.
0092. Bạn thử nghe lại từ bơn một lần nữa nhé.
0093. Bài học hôm nay có từ bếch ở ngay đầu câu.
0094. Từ bếch nằm ở giữa câu, không đọc quá nhanh.
0095. Bạn có thể đọc từ chặt to hơn một chút.
0096. Nếu đọc sai từ chặt thì đọc lại từ đầu.
0097. Hôm nay tôi muốn đọc từ cing thật rõ ràng.
0098. Từ cing nghe hơi lạ, nhưng rất dễ đọc.
0099. Trong câu này có từ nướt, đọc chậm thôi.
0100. Tôi sẽ đọc từ nướt ba lần cho bạn nghe.
0101. Tôi đã viết từ dượt lên bảng để bạn xem.
0102. Khi gặp từ dượt, bạn đừng ngắt giữa chừng.
0103. Chúng ta gặp từ đeo trong bài đọc hôm qua.
0104. Tôi đánh dấu từ đeo bằng bút đỏ.
0105. Mọi người cùng nhìn vào từ bì trên màn hình.
0106. Cuối buổi, tôi sẽ kiểm tra lại từ bì.
0107. Cái từ víp này xuất hiện khá ít trong sách.
0108. Bạn thử nghe lại từ víp một lần nữa nhé.
0109. Bài học hôm nay có từ toan ở ngay đầu câu.
0110. Từ toan nằm ở giữa câu, không đọc quá nhanh.
0111. Bạn có thể đọc từ toanh to hơn một chút.
0112. Nếu đọc sai từ toanh thì đọc lại từ đầu.
0113. Hôm nay tôi muốn đọc từ tua thật rõ ràng.
0114. Từ tua nghe hơi lạ, nhưng rất dễ đọc.
0115. Trong câu này có từ chẵn, đọc chậm thôi.
0116. Tôi sẽ đọc từ chẵn ba lần cho bạn nghe.
0117. Tôi đã viết từ gáy lên bảng để bạn xem.
0118. Khi gặp từ gáy, bạn đừng ngắt giữa chừng.
0119. Chúng ta gặp từ cầ trong bài đọc hôm qua.
0120. Tôi đánh dấu từ cầ bằng bút đỏ.
0121. Mọi người cùng nhìn vào từ đùng trên màn hình.
0122. Cuối buổi, tôi sẽ kiểm tra lại từ đùng.
0123. Cái từ lòi này xuất hiện khá ít trong sách.
0124. Bạn thử nghe lại từ lòi một lần nữa nhé.
0125. Bài học hôm nay có từ lếc ở ngay đầu câu.
0126. Từ lếc nằm ở giữa câu, không đọc quá nhanh.
0127. Bạn có thể đọc từ xộp to hơn một chút.
0128. Nếu đọc sai từ xộp thì đọc lại từ đầu.
0129. Hôm nay tôi muốn đọc từ lạn thật rõ ràng.
0130. Từ lạn nghe hơi lạ, nhưng rất dễ đọc.
0131. Trong câu này có từ uầ, đọc chậm thôi.
0132. Tôi sẽ đọc từ uầ ba lần cho bạn nghe.
0133. Tôi đã viết từ dam lên bảng để bạn xem.
0134. Khi gặp từ dam, bạn đừng ngắt giữa chừng.
0135. Chúng ta gặp từ nủng trong bài đọc hôm qua.
0136. Tôi đánh dấu từ nủng bằng bút đỏ.
0137. Mọi người cùng nhìn vào từ vong trên màn hình.
0138. Cuối buổi, tôi sẽ kiểm tra lại từ vong.
0139. Cái từ khoả này xuất hiện khá ít trong sách.
0140. Bạn thử nghe lại từ khoả một lần nữa nhé.
0141. Bài học hôm nay có từ củi ở ngay đầu câu.
0142. Từ củi nằm ở giữa câu, không đọc quá nhanh.
0143. Bạn có thể đọc từ đụng to hơn một chút.
0144. Nếu đọc sai từ đụng thì đọc lại từ đầu.
0145. Hôm nay tôi muốn đọc từ xóp thật rõ ràng.
0146. Từ xóp nghe hơi lạ, nhưng rất dễ đọc.
0147. Trong câu này có từ thuận, đọc chậm thôi.
0148. Tôi sẽ đọc từ thuận ba lần cho bạn nghe.
0149. Tôi đã viết từ dốt lên bảng để bạn xem.
0150. Khi gặp từ dốt, bạn đừng ngắt giữa chừng.
0151. Chúng ta gặp từ đẻ trong bài đọc hôm qua.
0152. Tôi đánh dấu từ đẻ bằng bút đỏ.
0153. Mọi người cùng nhìn vào từ mịt trên màn hình.
0154. Cuối buổi, tôi sẽ kiểm tra lại từ mịt.
0155. Cái từ kệp này xuất hiện khá ít trong sách.
0156. Bạn thử nghe lại từ kệp một lần nữa nhé.
0157. Bài học hôm nay có từ móng ở ngay đầu câu.
0158. Từ móng nằm ở giữa câu, không đọc quá nhanh.
0159. Bạn có thể đọc từ dăm to hơn một chút.
0160. Nếu đọc sai từ dăm thì đọc lại từ đầu.
0161. Hôm nay tôi muốn đọc từ thoả thật rõ ràng.
0162. Từ thoả nghe hơi lạ, nhưng rất dễ đọc.
0163. Trong câu này có từ mãn, đọc chậm thôi.
0164. Tôi sẽ đọc từ mãn ba lần cho bạn nghe.
0165. Tôi đã viết từ đoản lên bảng để bạn xem.
0166. Khi gặp từ đoản, bạn đừng ngắt giữa chừng.
0167. Chúng ta gặp từ ngoái trong bài đọc hôm qua.
0168. Tôi đánh dấu từ ngoái bằng bút đỏ.
0169. Mọi người cùng nhìn vào từ lông trên màn hình.
0170. Cuối buổi, tôi sẽ kiểm tra lại từ lông.
0171. Cái từ hiệp này xuất hiện khá ít trong sách.
0172. Bạn thử nghe lại từ hiệp một lần nữa nhé.
0173. Bài học hôm nay có từ loãng ở ngay đầu câu.
0174. Từ loãng nằm ở giữa câu, không đọc quá nhanh.
0175. Bạn có thể đọc từ đoan to hơn một chút.
0176. Nếu đọc sai từ đoan thì đọc lại từ đầu.
0177. Hôm nay tôi muốn đọc từ đùm thật rõ ràng.
0178. Từ đùm nghe hơi lạ, nhưng rất dễ đọc.
0179. Trong câu này có từ tông, đọc chậm thôi.
0180. Tôi sẽ đọc từ tông ba lần cho bạn nghe.
0181. Tôi đã viết từ kiếp lên bảng để bạn xem.
0182. Khi gặp từ kiếp, bạn đừng ngắt giữa chừng.
0183. Chúng ta gặp từ thạch trong bài đọc hôm qua.
0184. Tôi đánh dấu từ thạch bằng bút đỏ.
0185. Mọi người cùng nhìn vào từ quỷ trên màn hình.
0186. Cuối buổi, tôi sẽ kiểm tra lại từ quỷ.
0187. Cái từ nhãn này xuất hiện khá ít trong sách.
0188. Bạn thử nghe lại từ nhãn một lần nữa nhé.
0189. Bài học hôm nay có từ hi ở ngay đầu câu.
0190. Từ hi nằm ở giữa câu, không đọc quá nhanh.
0191. Bạn có thể đọc từ dou to hơn một chút.
0192. Nếu đọc sai từ dou thì đọc lại từ đầu.
0193. Hôm nay tôi muốn đọc từ nướn thật rõ ràng.
0194. Từ nướn nghe hơi lạ, nhưng rất dễ đọc.
0195. Trong câu này có từ tỉnh, đọc chậm thôi.
0196. Tôi sẽ đọc từ tỉnh ba lần cho bạn nghe.
0197. Tôi đã viết từ kỷ lên bảng để bạn xem.
0198. Khi gặp từ kỷ, bạn đừng ngắt giữa chừng.
0199. Chúng ta gặp từ loa trong bài đọc hôm qua.
0200. Tôi đánh dấu từ loa bằng bút đỏ.
0201. Mọi người cùng nhìn vào từ đau trên màn hình.
0202. Cuối buổi, tôi sẽ kiểm tra lại từ đau.
0203. Cái từ tráng này xuất hiện khá ít trong sách.
0204. Bạn thử nghe lại từ tráng một lần nữa nhé.
0205. Bài học hôm nay có từ sói ở ngay đầu câu.
0206. Từ sói nằm ở giữa câu, không đọc quá nhanh.
0207. Bạn có thể đọc từ diệu to hơn một chút.
0208. Nếu đọc sai từ diệu thì đọc lại từ đầu.
0209. Hôm nay tôi muốn đọc từ thấu thật rõ ràng.
0210. Từ thấu nghe hơi lạ, nhưng rất dễ đọc.
0211. Trong câu này có từ thiếc, đọc chậm thôi.
0212. Tôi sẽ đọc từ thiếc ba lần cho bạn nghe.
0213. Tôi đã viết từ nều lên bảng để bạn xem.
0214. Khi gặp từ nều, bạn đừng ngắt giữa chừng.
0215. Chúng ta gặp từ thịch trong bài đọc hôm qua.
0216. Tôi đánh dấu từ thịch bằng bút đỏ.
0217. Mọi người cùng nhìn vào từ dợ trên màn hình.
0218. Cuối buổi, tôi sẽ kiểm tra lại từ dợ.
0219. Cái từ vàn này xuất hiện khá ít trong sách.
0220. Bạn thử nghe lại từ vàn một lần nữa nhé.
0221. Bài học hôm nay có từ ổ ở ngay đầu câu.
0222. Từ ổ nằm ở giữa câu, không đọc quá nhanh.
0223. Bạn có thể đọc từ điềm to hơn một chút.
0224. Nếu đọc sai từ điềm thì đọc lại từ đầu.
0225. Hôm nay tôi muốn đọc từ tộn thật rõ ràng.
0226. Từ tộn nghe hơi lạ, nhưng rất dễ đọc.
0227. Trong câu này có từ gý, đọc chậm thôi.
0228. Tôi sẽ đọc từ gý ba lần cho bạn nghe.
0229. Tôi đã viết từ giặt lên bảng để bạn xem.
0230. Khi gặp từ giặt, bạn đừng ngắt giữa chừng.
0231. Chúng ta gặp từ nhiệu trong bài đọc hôm qua.
0232. Tôi đánh dấu từ nhiệu bằng bút đỏ.
0233. Mọi người cùng nhìn vào từ sing trên màn hình.
0234. Cuối buổi, tôi sẽ kiểm tra lại từ sing.
0235. Cái từ nang này xuất hiện khá ít trong sách.
0236. Bạn thử nghe lại từ nang một lần nữa nhé.
0237. Bài học hôm nay có từ máng ở ngay đầu câu.
0238. Từ máng nằm ở giữa câu, không đọc quá nhanh.
0239. Bạn có thể đọc từ hập to hơn một chút.
0240. Nếu đọc sai từ hập thì đọc lại từ đầu.
0241. Hôm nay tôi muốn đọc từ đói thật rõ ràng.
0242. Từ đói nghe hơi lạ, nhưng rất dễ đọc.
0243. Trong câu này có từ điệp, đọc chậm thôi.
0244. Tôi sẽ đọc từ điệp ba lần cho bạn nghe.
0245. Tôi đã viết từ dõ lên bảng để bạn xem.
0246. Khi gặp từ dõ, bạn đừng ngắt giữa chừng.
0247. Chúng ta gặp từ dí trong bài đọc hôm qua.
0248. Tôi đánh dấu từ dí bằng bút đỏ.
0249. Mọi người cùng nhìn vào từ treo trên màn hình.
0250. Cuối buổi, tôi sẽ kiểm tra lại từ treo.
0251. Cái từ đúc này xuất hiện khá ít trong sách.
0252. Bạn thử nghe lại từ đúc một lần nữa nhé.
0253. Bài học hôm nay có từ buông ở ngay đầu câu.
0254. Từ buông nằm ở giữa câu, không đọc quá nhanh.
0255. Bạn có thể đọc từ lỏng to hơn một chút.
0256. Nếu đọc sai từ lỏng thì đọc lại từ đầu.
0257. Hôm nay tôi muốn đọc từ hỏng thật rõ ràng.
0258. Từ hỏng nghe hơi lạ, nhưng rất dễ đọc.
0259. Trong câu này có từ ngỡ, đọc chậm thôi.
0260. Tôi sẽ đọc từ ngỡ ba lần cho bạn nghe.
0261. Tôi đã viết từ ngẩn lên bảng để bạn xem.
0262. Khi gặp từ ngẩn, bạn đừng ngắt giữa chừng.
0263. Chúng ta gặp từ tràn trong bài đọc hôm qua.
0264. Tôi đánh dấu từ tràn bằng bút đỏ.
0265. Mọi người cùng nhìn vào từ nan trên màn hình.
0266. Cuối buổi, tôi sẽ kiểm tra lại từ nan.
0267. Cái từ ngoạn này xuất hiện khá ít trong sách.
0268. Bạn thử nghe lại từ ngoạn một lần nữa nhé.
0269. Bài học hôm nay có từ chuộng ở ngay đầu câu.
0270. Từ chuộng nằm ở giữa câu, không đọc quá nhanh.
0271. Bạn có thể đọc từ hổ to hơn một chút.
0272. Nếu đọc sai từ hổ thì đọc lại từ đầu.
0273. Hôm nay tôi muốn đọc từ nhằm thật rõ ràng.
0274. Từ nhằm nghe hơi lạ, nhưng rất dễ đọc.
0275. Trong câu này có từ mâu, đọc chậm thôi.
0276. Tôi sẽ đọc từ mâu ba lần cho bạn nghe.
0277. Tôi đã viết từ thuẫn lên bảng để bạn xem.
0278. Khi gặp từ thuẫn, bạn đừng ngắt giữa chừng.
0279. Chúng ta gặp từ rắc trong bài đọc hôm qua.
0280. Tôi đánh dấu từ rắc bằng bút đỏ.
0281. Mọi người cùng nhìn vào từ rớ trên màn hình.
0282. Cuối buổi, tôi sẽ kiểm tra lại từ rớ.
0283. Cái từ ngình này xuất hiện khá ít trong sách.
0284. Bạn thử nghe lại từ ngình một lần nữa nhé.
0285. Bài học hôm nay có từ mẻ ở ngay đầu câu.
0286. Từ mẻ nằm ở giữa câu, không đọc quá nhanh.
0287. Bạn có thể đọc từ dọm to hơn một chút.
0288. Nếu đọc sai từ dọm thì đọc lại từ đầu.
0289. Hôm nay tôi muốn đọc từ kụ thật rõ ràng.
0290. Từ kụ nghe hơi lạ, nhưng rất dễ đọc.
0291. Trong câu này có từ bão, đọc chậm thôi.
0292. Tôi sẽ đọc từ bão ba lần cho bạn nghe.
0293. Tôi đã viết từ thụt lên bảng để bạn xem.
0294. Khi gặp từ thụt, bạn đừng ngắt giữa chừng.
0295. Chúng ta gặp từ nhì trong bài đọc hôm qua.
0296. Tôi đánh dấu từ nhì bằng bút đỏ.
0297. Mọi người cùng nhìn vào từ héo trên màn hình.
0298. Cuối buổi, tôi sẽ kiểm tra lại từ héo.
0299. Cái từ tưới này xuất hiện khá ít trong sách.
0300. Bạn thử nghe lại từ tưới một lần nữa nhé.
0301. Bài học hôm nay có từ phóc ở ngay đầu câu.
0302. Từ phóc nằm ở giữa câu, không đọc quá nhanh.
0303. Bạn có thể đọc từ chàm to hơn một chút.
0304. Nếu đọc sai từ chàm thì đọc lại từ đầu.
0305. Hôm nay tôi muốn đọc từ đỉnh thật rõ ràng.
0306. Từ đỉnh nghe hơi lạ, nhưng rất dễ đọc.
0307. Trong câu này có từ núp, đọc chậm thôi.
0308. Tôi sẽ đọc từ núp ba lần cho bạn nghe.
0309. Tôi đã viết từ trự lên bảng để bạn xem.
0310. Khi gặp từ trự, bạn đừng ngắt giữa chừng.
0311. Chúng ta gặp từ sanh trong bài đọc hôm qua.
0312. Tôi đánh dấu từ sanh bằng bút đỏ.
0313. Mọi người cùng nhìn vào từ mỉ trên màn hình.
0314. Cuối buổi, tôi sẽ kiểm tra lại từ mỉ.
0315. Cái từ điên này xuất hiện khá ít trong sách.
0316. Bạn thử nghe lại từ điên một lần nữa nhé.
0317. Bài học hôm nay có từ hừm ở ngay đầu câu.
0318. Từ hừm nằm ở giữa câu, không đọc quá nhanh.
0319. Bạn có thể đọc từ xỉ to hơn một chút.
0320. Nếu đọc sai từ xỉ thì đọc lại từ đầu.
0321. Hôm nay tôi muốn đọc từ trừu thật rõ ràng.
0322. Từ trừu nghe hơi lạ, nhưng rất dễ đọc.
0323. Trong câu này có từ trưng, đọc chậm thôi.
0324. Tôi sẽ đọc từ trưng ba lần cho bạn nghe.
0325. Tôi đã viết từ nấu lên bảng để bạn xem.
0326. Khi gặp từ nấu, bạn đừng ngắt giữa chừng.
0327. Chúng ta gặp từ răng trong bài đọc hôm qua.
0328. Tôi đánh dấu từ răng bằng bút đỏ.
0329. Mọi người cùng nhìn vào từ cưa trên màn hình.
0330. Cuối buổi, tôi sẽ kiểm tra lại từ cưa.
0331. Cái từ loạng này xuất hiện khá ít trong sách.
0332. Bạn thử nghe lại từ loạng một lần nữa nhé.
0333. Bài học hôm nay có từ đầ ở ngay đầu câu.
0334. Từ đầ nằm ở giữa câu, không đọc quá nhanh.
0335. Bạn có thể đọc từ bày to hơn một chút.
0336. Nếu đọc sai từ bày thì đọc lại từ đầu.
0337. Hôm nay tôi muốn đọc từ nế thật rõ ràng.
0338. Từ nế nghe hơi lạ, nhưng rất dễ đọc.
0339. Trong câu này có từ dệnh, đọc chậm thôi.
0340. Tôi sẽ đọc từ dệnh ba lần cho bạn nghe.
0341. Tôi đã viết từ lớt lên bảng để bạn xem.
0342. Khi gặp từ lớt, bạn đừng ngắt giữa chừng.
0343. Chúng ta gặp từ chăn trong bài đọc hôm qua.
0344. Tôi đánh dấu từ chăn bằng bút đỏ.
0345. Mọi người cùng nhìn vào từ dục trên màn hình.
0346. Cuối buổi, tôi sẽ kiểm tra lại từ dục.
0347. Cái từ ngã này xuất hiện khá ít trong sách.
0348. Bạn thử nghe lại từ ngã một lần nữa nhé.
0349. Bài học hôm nay có từ ngoạt ở ngay đầu câu.
0350. Từ ngoạt nằm ở giữa câu, không đọc quá nhanh.
0351. Bạn có thể đọc từ suốt to hơn một chút.
0352. Nếu đọc sai từ suốt thì đọc lại từ đầu.
0353. Hôm nay tôi muốn đọc từ rỡ thật rõ ràng.
0354. Từ rỡ nghe hơi lạ, nhưng rất dễ đọc.
0355. Trong câu này có từ phươ, đọc chậm thôi.
0356. Tôi sẽ đọc từ phươ ba lần cho bạn nghe.
0357. Tôi đã viết từ dẽ lên bảng để bạn xem.
0358. Khi gặp từ dẽ, bạn đừng ngắt giữa chừng.
0359. Chúng ta gặp từ thổi trong bài đọc hôm qua.
0360. Tôi đánh dấu từ thổi bằng bút đỏ.
0361. Mọi người cùng nhìn vào từ thượng trên màn hình.
0362. Cuối buổi, tôi sẽ kiểm tra lại từ thượng.
0363. Cái từ nộn này xuất hiện khá ít trong sách.
0364. Bạn thử nghe lại từ nộn một lần nữa nhé.
0365. Bài học hôm nay có từ gắm ở ngay đầu câu.
0366. Từ gắm nằm ở giữa câu, không đọc quá nhanh.
0367. Bạn có thể đọc từ bến to hơn một chút.
0368. Nếu đọc sai từ bến thì đọc lại từ đầu.
0369. Hôm nay tôi muốn đọc từ phảm thật rõ ràng.
0370. Từ phảm nghe hơi lạ, nhưng rất dễ đọc.
0371. Trong câu này có từ chù, đọc chậm thôi.
0372. Tôi sẽ đọc từ chù ba lần cho bạn nghe.
0373. Tôi đã viết từ ngyễn lên bảng để bạn xem.
0374. Khi gặp từ ngyễn, bạn đừng ngắt giữa chừng.
0375. Chúng ta gặp từ ngyện trong bài đọc hôm qua.
0376. Tôi đánh dấu từ ngyện bằng bút đỏ.
0377. Mọi người cùng nhìn vào từ hỏ trên màn hình.
0378. Cuối buổi, tôi sẽ kiểm tra lại từ hỏ.
0379. Cái từ trân này xuất hiện khá ít trong sách.
0380. Bạn thử nghe lại từ trân một lần nữa nhé.
0381. Bài học hôm nay có từ lãng ở ngay đầu câu.
0382. Từ lãng nằm ở giữa câu, không đọc quá nhanh.
0383. Bạn có thể đọc từ băn to hơn một chút.
0384. Nếu đọc sai từ băn thì đọc lại từ đầu.
0385. Hôm nay tôi muốn đọc từ ủi thật rõ ràng.
0386. Từ ủi nghe hơi lạ, nhưng rất dễ đọc.
0387. Trong câu này có từ mưa, đọc chậm thôi.
0388. Tôi sẽ đọc từ mưa ba lần cho bạn nghe.
0389. Tôi đã viết từ gió lên bảng để bạn xem.
0390. Khi gặp từ gió, bạn đừng ngắt giữa chừng.
0391. Chúng ta gặp từ đành trong bài đọc hôm qua.
0392. Tôi đánh dấu từ đành bằng bút đỏ.
0393. Mọi người cùng nhìn vào từ phận trên màn hình.
0394. Cuối buổi, tôi sẽ kiểm tra lại từ phận.
0395. Cái từ chạt này xuất hiện khá ít trong sách.
0396. Bạn thử nghe lại từ chạt một lần nữa nhé.
0397. Bài học hôm nay có từ khỏ ở ngay đầu câu.
0398. Từ khỏ nằm ở giữa câu, không đọc quá nhanh.
0399. Bạn có thể đọc từ phé to hơn một chút.
0400. Nếu đọc sai từ phé thì đọc lại từ đầu.
0401. Hôm nay tôi muốn đọc từ kện thật rõ ràng.
0402. Từ kện nghe hơi lạ, nhưng rất dễ đọc.
0403. Trong câu này có từ ngộp, đọc chậm thôi.
0404. Tôi sẽ đọc từ ngộp ba lần cho bạn nghe.
0405. Tôi đã viết từ ben lên bảng để bạn xem.
0406. Khi gặp từ ben, bạn đừng ngắt giữa chừng.
0407. Chúng ta gặp từ sấy trong bài đọc hôm qua.
0408. Tôi đánh dấu từ sấy bằng bút đỏ.
0409. Mọi người cùng nhìn vào từ đôn trên màn hình.
0410. Cuối buổi, tôi sẽ kiểm tra lại từ đôn.
0411. Cái từ ming này xuất hiện khá ít trong sách.
0412. Bạn thử nghe lại từ ming một lần nữa nhé.
0413. Bài học hôm nay có từ cút ở ngay đầu câu.
0414. Từ cút nằm ở giữa câu, không đọc quá nhanh.
0415. Bạn có thể đọc từ hoang to hơn một chút.
0416. Nếu đọc sai từ hoang thì đọc lại từ đầu.
0417. Hôm nay tôi muốn đọc từ guy thật rõ ràng.
0418. Từ guy nghe hơi lạ, nhưng rất dễ đọc.
0419. Trong câu này có từ hời, đọc chậm thôi.
0420. Tôi sẽ đọc từ hời ba lần cho bạn nghe.
0421. Tôi đã viết từ thò lên bảng để bạn xem.
0422. Khi gặp từ thò, bạn đừng ngắt giữa chừng.
0423. Chúng ta gặp từ kiể trong bài đọc hôm qua.
0424. Tôi đánh dấu từ kiể bằng bút đỏ.
0425. Mọi người cùng nhìn vào từ tùm trên màn hình.
0426. Cuối buổi, tôi sẽ kiểm tra lại từ tùm.
0427. Cái từ hí này xuất hiện khá ít trong sách.
0428. Bạn thử nghe lại từ hí một lần nữa nhé.
0429. Bài học hôm nay có từ què ở ngay đầu câu.
0430. Từ què nằm ở giữa câu, không đọc quá nhanh.
0431. Bạn có thể đọc từ thằ to hơn một chút.
0432. Nếu đọc sai từ thằ thì đọc lại từ đầu.
0433. Hôm nay tôi muốn đọc từ cuồng thật rõ ràng.
0434. Từ cuồng nghe hơi lạ, nhưng rất dễ đọc.
0435. Trong câu này có từ vịt, đọc chậm thôi.
0436. Tôi sẽ đọc từ vịt ba lần cho bạn nghe.
0437. Tôi đã viết từ còng lên bảng để bạn xem.
0438. Khi gặp từ còng, bạn đừng ngắt giữa chừng.
0439. Chúng ta gặp từ théo trong bài đọc hôm qua.
0440. Tôi đánh dấu từ théo bằng bút đỏ.
0441. Mọi người cùng nhìn vào từ soi trên màn hình.
0442. Cuối buổi, tôi sẽ kiểm tra lại từ soi.
0443. Cái từ ping này xuất hiện khá ít trong sách.
0444. Bạn thử nghe lại từ ping một lần nữa nhé.
0445. Bài học hôm nay có từ uống ở ngay đầu câu.
0446. Từ uống nằm ở giữa câu, không đọc quá nhanh.
0447. Bạn có thể đọc từ duyệu to hơn một chút.
0448. Nếu đọc sai từ duyệu thì đọc lại từ đầu.
0449. Hôm nay tôi muốn đọc từ phia thật rõ ràng.
0450. Từ phia nghe hơi lạ, nhưng rất dễ đọc.
0451. Trong câu này có từ săn, đọc chậm thôi.
0452. Tôi sẽ đọc từ săn ba lần cho bạn nghe.
0453. Tôi đã viết từ thuếc lên bảng để bạn xem.
0454. Khi gặp từ thuếc, bạn đừng ngắt giữa chừng.
0455. Chúng ta gặp từ giỏ trong bài đọc hôm qua.
0456. Tôi đánh dấu từ giỏ bằng bút đỏ.
0457. Mọi người cùng nhìn vào từ toen trên màn hình.
0458. Cuối buổi, tôi sẽ kiểm tra lại từ toen.
0459. Cái từ hu này xuất hiện khá ít trong sách.
0460. Bạn thử nghe lại từ hu một lần nữa nhé.
0461. Bài học hôm nay có từ ang ở ngay đầu câu.
0462. Từ ang nằm ở giữa câu, không đọc quá nhanh.
0463. Bạn có thể đọc từ nguyền to hơn một chút.
0464. Nếu đọc sai từ nguyền thì đọc lại từ đầu.
0465. Hôm nay tôi muốn đọc từ phơ thật rõ ràng.
0466. Từ phơ nghe hơi lạ, nhưng rất dễ đọc.
0467. Trong câu này có từ lam, đọc chậm thôi.
0468. Tôi sẽ đọc từ lam ba lần cho bạn nghe.
0469. Tôi đã viết từ rát lên bảng để bạn xem.
0470. Khi gặp từ rát, bạn đừng ngắt giữa chừng.
0471. Chúng ta gặp từ phà trong bài đọc hôm qua.
0472. Tôi đánh dấu từ phà bằng bút đỏ.
0473. Mọi người cùng nhìn vào từ chio trên màn hình.
0474. Cuối buổi, tôi sẽ kiểm tra lại từ chio.
0475. Cái từ thươ này xuất hiện khá ít trong sách.
0476. Bạn thử nghe lại từ thươ một lần nữa nhé.
0477. Bài học hôm nay có từ gạo ở ngay đầu câu.
0478. Từ gạo nằm ở giữa câu, không đọc quá nhanh.
0479. Bạn có thể đọc từ giục to hơn một chút.
0480. Nếu đọc sai từ giục thì đọc lại từ đầu.
0481. Hôm nay tôi muốn đọc từ lận thật rõ ràng.
0482. Từ lận nghe hơi lạ, nhưng rất dễ đọc.
0483. Trong câu này có từ miên, đọc chậm thôi.
0484. Tôi sẽ đọc từ miên ba lần cho bạn nghe.
0485. Tôi đã viết từ chợ lên bảng để bạn xem.
0486. Khi gặp từ chợ, bạn đừng ngắt giữa chừng.
0487. Chúng ta gặp từ bền trong bài đọc hôm qua.
0488. Tôi đánh dấu từ bền bằng bút đỏ.
0489. Mọi người cùng nhìn vào từ xứ trên màn hình.
0490. Cuối buổi, tôi sẽ kiểm tra lại từ xứ.
0491. Cái từ dịp này xuất hiện khá ít trong sách.
0492. Bạn thử nghe lại từ dịp một lần nữa nhé.
0493. Bài học hôm nay có từ ce ở ngay đầu câu.
0494. Từ ce nằm ở giữa câu, không đọc quá nhanh.
0495. Bạn có thể đọc từ vã to hơn một chút.
0496. Nếu đọc sai từ vã thì đọc lại từ đầu.
0497. Hôm nay tôi muốn đọc từ moo thật rõ ràng.
0498. Từ moo nghe hơi lạ, nhưng rất dễ đọc.
0499. Trong câu này có từ lõi, đọc chậm thôi.
0500. Tôi sẽ đọc từ lõi ba lần cho bạn nghe.
0501. Tôi đã viết từ vướng lên bảng để bạn xem.
0502. Khi gặp từ vướng, bạn đừng ngắt giữa chừng.
0503. Chúng ta gặp từ phủ trong bài đọc hôm qua.
0504. Tôi đánh dấu từ phủ bằng bút đỏ.
0505. Mọi người cùng nhìn vào từ tiếm trên màn hình.
0506. Cuối buổi, tôi sẽ kiểm tra lại từ tiếm.
0507. Cái từ huống này xuất hiện khá ít trong sách.
0508. Bạn thử nghe lại từ huống một lần nữa nhé.
0509. Bài học hôm nay có từ gắt ở ngay đầu câu.
0510. Từ gắt nằm ở giữa câu, không đọc quá nhanh.
0511. Bạn có thể đọc từ toản to hơn một chút.
0512. Nếu đọc sai từ toản thì đọc lại từ đầu.
0513. Hôm nay tôi muốn đọc từ cức thật rõ ràng.
0514. Từ cức nghe hơi lạ, nhưng rất dễ đọc.
0515. Trong câu này có từ bậy, đọc chậm thôi.
0516. Tôi sẽ đọc từ bậy ba lần cho bạn nghe.
0517. Tôi đã viết từ hút lên bảng để bạn xem.
0518. Khi gặp từ hút, bạn đừng ngắt giữa chừng.
0519. Chúng ta gặp từ nắp trong bài đọc hôm qua.
0520. Tôi đánh dấu từ nắp bằng bút đỏ.
0521. Mọi người cùng nhìn vào từ dấy trên màn hình.
0522. Cuối buổi, tôi sẽ kiểm tra lại từ dấy.
0523. Cái từ mề này xuất hiện khá ít trong sách.
0524. Bạn thử nghe lại từ mề một lần nữa nhé.
0525. Bài học hôm nay có từ đạch ở ngay đầu câu.
0526. Từ đạch nằm ở giữa câu, không đọc quá nhanh.
0527. Bạn có thể đọc từ nẻ to hơn một chút.
0528. Nếu đọc sai từ nẻ thì đọc lại từ đầu.
0529. Hôm nay tôi muốn đọc từ nũ thật rõ ràng.
0530. Từ nũ nghe hơi lạ, nhưng rất dễ đọc.
0531. Trong câu này có từ nụt, đọc chậm thôi.
0532. Tôi sẽ đọc từ nụt ba lần cho bạn nghe.
0533. Tôi đã viết từ oẳng lên bảng để bạn xem.
0534. Khi gặp từ oẳng, bạn đừng ngắt giữa chừng.
0535. Chúng ta gặp từ thấm trong bài đọc hôm qua.
0536. Tôi đánh dấu từ thấm bằng bút đỏ.
0537. Mọi người cùng nhìn vào từ khẹn trên màn hình.
0538. Cuối buổi, tôi sẽ kiểm tra lại từ khẹn.
0539. Cái từ lân này xuất hiện khá ít trong sách.
0540. Bạn thử nghe lại từ lân một lần nữa nhé.
0541. Bài học hôm nay có từ há ở ngay đầu câu.
0542. Từ há nằm ở giữa câu, không đọc quá nhanh.
0543. Bạn có thể đọc từ rụng to hơn một chút.
0544. Nếu đọc sai từ rụng thì đọc lại từ đầu.
0545. Hôm nay tôi muốn đọc từ thểo thật rõ ràng.
0546. Từ thểo nghe hơi lạ, nhưng rất dễ đọc.
0547. Trong câu này có từ nện, đọc chậm thôi.
0548. Tôi sẽ đọc từ nện ba lần cho bạn nghe.
0549. Tôi đã viết từ đở lên bảng để bạn xem.
0550. Khi gặp từ đở, bạn đừng ngắt giữa chừng.
0551. Chúng ta gặp từ nháp trong bài đọc hôm qua.
0552. Tôi đánh dấu từ nháp bằng bút đỏ.
0553. Mọi người cùng nhìn vào từ nghẽn trên màn hình.
0554. Cuối buổi, tôi sẽ kiểm tra lại từ nghẽn.
0555. Cái từ tuổi này xuất hiện khá ít trong sách.
0556. Bạn thử nghe lại từ tuổi một lần nữa nhé.
0557. Bài học hôm nay có từ nòng ở ngay đầu câu.
0558. Từ nòng nằm ở giữa câu, không đọc quá nhanh.
0559. Bạn có thể đọc từ huyết to hơn một chút.
0560. Nếu đọc sai từ huyết thì đọc lại từ đầu.
0561. Hôm nay tôi muốn đọc từ phấn thật rõ ràng.
0562. Từ phấn nghe hơi lạ, nhưng rất dễ đọc.
0563. Trong câu này có từ rầm, đọc chậm thôi.
0564. Tôi sẽ đọc từ rầm ba lần cho bạn nghe.
0565. Tôi đã viết từ rộ lên bảng để bạn xem.
0566. Khi gặp từ rộ, bạn đừng ngắt giữa chừng.
0567. Chúng ta gặp từ phên trong bài đọc hôm qua.
0568. Tôi đánh dấu từ phên bằng bút đỏ.
0569. Mọi người cùng nhìn vào từ thê trên màn hình.
0570. Cuối buổi, tôi sẽ kiểm tra lại từ thê.
0571. Cái từ cấm này xuất hiện khá ít trong sách.
0572. Bạn thử nghe lại từ cấm một lần nữa nhé.
0573. Bài học hôm nay có từ chệt ở ngay đầu câu.
0574. Từ chệt nằm ở giữa câu, không đọc quá nhanh.
0575. Bạn có thể đọc từ giành to hơn một chút.
0576. Nếu đọc sai từ giành thì đọc lại từ đầu.
0577. Hôm nay tôi muốn đọc từ trơn thật rõ ràng.
0578. Từ trơn nghe hơi lạ, nhưng rất dễ đọc.
0579. Trong câu này có từ chểm, đọc chậm thôi.
0580. Tôi sẽ đọc từ chểm ba lần cho bạn nghe.
0581. Tôi đã viết từ pem lên bảng để bạn xem.
0582. Khi gặp từ pem, bạn đừng ngắt giữa chừng.
0583. Chúng ta gặp từ trẻ trong bài đọc hôm qua.
0584. Tôi đánh dấu từ trẻ bằng bút đỏ.
0585. Mọi người cùng nhìn vào từ nghèo trên màn hình.
0586. Cuối buổi, tôi sẽ kiểm tra lại từ nghèo.
0587. Cái từ bở này xuất hiện khá ít trong sách.
0588. Bạn thử nghe lại từ bở một lần nữa nhé.
0589. Bài học hôm nay có từ rủ ở ngay đầu câu.
0590. Từ rủ nằm ở giữa câu, không đọc quá nhanh.
0591. Bạn có thể đọc từ khắc to hơn một chút.
0592. Nếu đọc sai từ khắc thì đọc lại từ đầu.
0593. Hôm nay tôi muốn đọc từ phẩm thật rõ ràng.
0594. Từ phẩm nghe hơi lạ, nhưng rất dễ đọc.
0595. Trong câu này có từ dâ, đọc chậm thôi.
0596. Tôi sẽ đọc từ dâ ba lần cho bạn nghe.
0597. Tôi đã viết từ mồn lên bảng để bạn xem.
0598. Khi gặp từ mồn, bạn đừng ngắt giữa chừng.
0599. Chúng ta gặp từ trài trong bài đọc hôm qua.
0600. Tôi đánh dấu từ trài bằng bút đỏ.
0601. Mọi người cùng nhìn vào từ ngừng trên màn hình.
0602. Cuối buổi, tôi sẽ kiểm tra lại từ ngừng.
0603. Cái từ xấu này xuất hiện khá ít trong sách.
0604. Bạn thử nghe lại từ xấu một lần nữa nhé.
0605. Bài học hôm nay có từ sực ở ngay đầu câu.
0606. Từ sực nằm ở giữa câu, không đọc quá nhanh.
0607. Bạn có thể đọc từ bể to hơn một chút.
0608. Nếu đọc sai từ bể thì đọc lại từ đầu.
0609. Hôm nay tôi muốn đọc từ bơi thật rõ ràng.
0610. Từ bơi nghe hơi lạ, nhưng rất dễ đọc.
0611. Trong câu này có từ khét, đọc chậm thôi.
0612. Tôi sẽ đọc từ khét ba lần cho bạn nghe.
0613. Tôi đã viết từ binh lên bảng để bạn xem.
0614. Khi gặp từ binh, bạn đừng ngắt giữa chừng.
0615. Chúng ta gặp từ nóng trong bài đọc hôm qua.
0616. Tôi đánh dấu từ nóng bằng bút đỏ.
0617. Mọi người cùng nhìn vào từ nòi trên màn hình.
0618. Cuối buổi, tôi sẽ kiểm tra lại từ nòi.
0619. Cái từ hú này xuất hiện khá ít trong sách.
0620. Bạn thử nghe lại từ hú một lần nữa nhé.
0621. Bài học hôm nay có từ cóp ở ngay đầu câu.
0622. Từ cóp nằm ở giữa câu, không đọc quá nhanh.
0623. Bạn có thể đọc từ nhần to hơn một chút.
0624. Nếu đọc sai từ nhần thì đọc lại từ đầu.
0625. Hôm nay tôi muốn đọc từ phúc thật rõ ràng.
0626. Từ phúc nghe hơi lạ, nhưng rất dễ đọc.
0627. Trong câu này có từ nạn, đọc chậm thôi.
0628. Tôi sẽ đọc từ nạn ba lần cho bạn nghe.
0629. Tôi đã viết từ quyên lên bảng để bạn xem.
0630. Khi gặp từ quyên, bạn đừng ngắt giữa chừng.
0631. Chúng ta gặp từ đài trong bài đọc hôm qua.
0632. Tôi đánh dấu từ đài bằng bút đỏ.
0633. Mọi người cùng nhìn vào từ nhài trên màn hình.
0634. Cuối buổi, tôi sẽ kiểm tra lại từ nhài.
0635. Cái từ hiếm này xuất hiện khá ít trong sách.
0636. Bạn thử nghe lại từ hiếm một lần nữa nhé.
0637. Bài học hôm nay có từ túp ở ngay đầu câu.
0638. Từ túp nằm ở giữa câu, không đọc quá nhanh.
0639. Bạn có thể đọc từ đục to hơn một chút.
0640. Nếu đọc sai từ đục thì đọc lại từ đầu.
0641. Hôm nay tôi muốn đọc từ chọc thật rõ ràng.
0642. Từ chọc nghe hơi lạ, nhưng rất dễ đọc.
0643. Trong câu này có từ đâm, đọc chậm thôi.
0644. Tôi sẽ đọc từ đâm ba lần cho bạn nghe.
0645. Tôi đã viết từ hật lên bảng để bạn xem.
0646. Khi gặp từ hật, bạn đừng ngắt giữa chừng.
0647. Chúng ta gặp từ tậm trong bài đọc hôm qua.
0648. Tôi đánh dấu từ tậm bằng bút đỏ.
0649. Mọi người cùng nhìn vào từ nừa trên màn hình.
0650. Cuối buổi, tôi sẽ kiểm tra lại từ nừa.
0651. Cái từ hoặ này xuất hiện khá ít trong sách.
0652. Bạn thử nghe lại từ hoặ một lần nữa nhé.
0653. Bài học hôm nay có từ thoá ở ngay đầu câu.
0654. Từ thoá nằm ở giữa câu, không đọc quá nhanh.
0655. Bạn có thể đọc từ xí to hơn một chút.
0656. Nếu đọc sai từ xí thì đọc lại từ đầu.
0657. Hôm nay tôi muốn đọc từ hợi thật rõ ràng.
0658. Từ hợi nghe hơi lạ, nhưng rất dễ đọc.
0659. Trong câu này có từ bả, đọc chậm thôi.
0660. Tôi sẽ đọc từ bả ba lần cho bạn nghe.
0661. Tôi đã viết từ đing lên bảng để bạn xem.
0662. Khi gặp từ đing, bạn đừng ngắt giữa chừng.
0663. Chúng ta gặp từ bóp trong bài đọc hôm qua.
0664. Tôi đánh dấu từ bóp bằng bút đỏ.
0665. Mọi người cùng nhìn vào từ rả trên màn hình.
0666. Cuối buổi, tôi sẽ kiểm tra lại từ rả.
0667. Cái từ gặng này xuất hiện khá ít trong sách.
0668. Bạn thử nghe lại từ gặng một lần nữa nhé.
0669. Bài học hôm nay có từ só ở ngay đầu câu.
0670. Từ só nằm ở giữa câu, không đọc quá nhanh.
0671. Bạn có thể đọc từ phọc to hơn một chút.
0672. Nếu đọc sai từ phọc thì đọc lại từ đầu.
0673. Hôm nay tôi muốn đọc từ hãng thật rõ ràng.
0674. Từ hãng nghe hơi lạ, nhưng rất dễ đọc.
0675. Trong câu này có từ xai, đọc chậm thôi.
0676. Tôi sẽ đọc từ xai ba lần cho bạn nghe.
0677. Tôi đã viết từ nột lên bảng để bạn xem.
0678. Khi gặp từ nột, bạn đừng ngắt giữa chừng.
0679. Chúng ta gặp từ đíp trong bài đọc hôm qua.
0680. Tôi đánh dấu từ đíp bằng bút đỏ.
0681. Mọi người cùng nhìn vào từ hạ trên màn hình.
0682. Cuối buổi, tôi sẽ kiểm tra lại từ hạ.
0683. Cái từ dắm này xuất hiện khá ít trong sách.
0684. Bạn thử nghe lại từ dắm một lần nữa nhé.
0685. Bài học hôm nay có từ đoảng ở ngay đầu câu.
0686. Từ đoảng nằm ở giữa câu, không đọc quá nhanh.
0687. Bạn có thể đọc từ độc to hơn một chút.
0688. Nếu đọc sai từ độc thì đọc lại từ đầu.
0689. Hôm nay tôi muốn đọc từ trăng thật rõ ràng.
0690. Từ trăng nghe hơi lạ, nhưng rất dễ đọc.
0691. Trong câu này có từ hắn, đọc chậm thôi.
0692. Tôi sẽ đọc từ hắn ba lần cho bạn nghe.
0693. Tôi đã viết từ trội lên bảng để bạn xem.
0694. Khi gặp từ trội, bạn đừng ngắt giữa chừng.
0695. Chúng ta gặp từ tuện trong bài đọc hôm qua.
0696. Tôi đánh dấu từ tuện bằng bút đỏ.
0697. Mọi người cùng nhìn vào từ chỗi trên màn hình.
0698. Cuối buổi, tôi sẽ kiểm tra lại từ chỗi.
0699. Cái từ trễ này xuất hiện khá ít trong sách.
0700. Bạn thử nghe lại từ trễ một lần nữa nhé.
0701. Bài học hôm nay có từ lạt ở ngay đầu câu.
0702. Từ lạt nằm ở giữa câu, không đọc quá nhanh.
0703. Bạn có thể đọc từ hoán to hơn một chút.
0704. Nếu đọc sai từ hoán thì đọc lại từ đầu.
0705. Hôm nay tôi muốn đọc từ thoạt thật rõ ràng.
0706. Từ thoạt nghe hơi lạ, nhưng rất dễ đọc.
0707. Trong câu này có từ bẩn, đọc chậm thôi.
0708. Tôi sẽ đọc từ bẩn ba lần cho bạn nghe.
0709. Tôi đã viết từ quăng lên bảng để bạn xem.
0710. Khi gặp từ quăng, bạn đừng ngắt giữa chừng.
0711. Chúng ta gặp từ hỡi trong bài đọc hôm qua.
0712. Tôi đánh dấu từ hỡi bằng bút đỏ.
0713. Mọi người cùng nhìn vào từ nời trên màn hình.
0714. Cuối buổi, tôi sẽ kiểm tra lại từ nời.
0715. Cái từ dũng này xuất hiện khá ít trong sách.
0716. Bạn thử nghe lại từ dũng một lần nữa nhé.
0717. Bài học hôm nay có từ nhường ở ngay đầu câu.
0718. Từ nhường nằm ở giữa câu, không đọc quá nhanh.
0719. Bạn có thể đọc từ thịnh to hơn một chút.
0720. Nếu đọc sai từ thịnh thì đọc lại từ đầu.
0721. Hôm nay tôi muốn đọc từ vượng thật rõ ràng.
0722. Từ vượng nghe hơi lạ, nhưng rất dễ đọc.
0723. Trong câu này có từ triều, đọc chậm thôi.
0724. Tôi sẽ đọc từ triều ba lần cho bạn nghe.
0725. Tôi đã viết từ kiêu lên bảng để bạn xem.
0726. Khi gặp từ kiêu, bạn đừng ngắt giữa chừng.
0727. Chúng ta gặp từ nạo trong bài đọc hôm qua.
0728. Tôi đánh dấu từ nạo bằng bút đỏ.
0729. Mọi người cùng nhìn vào từ ách trên màn hình.
0730. Cuối buổi, tôi sẽ kiểm tra lại từ ách.
0731. Cái từ ngai này xuất hiện khá ít trong sách.
0732. Bạn thử nghe lại từ ngai một lần nữa nhé.
0733. Bài học hôm nay có từ mạo ở ngay đầu câu.
0734. Từ mạo nằm ở giữa câu, không đọc quá nhanh.
0735. Bạn có thể đọc từ sò to hơn một chút.
0736. Nếu đọc sai từ sò thì đọc lại từ đầu.
0737. Hôm nay tôi muốn đọc từ quỳ thật rõ ràng.
0738. Từ quỳ nghe hơi lạ, nhưng rất dễ đọc.
0739. Trong câu này có từ gối, đọc chậm thôi.
0740. Tôi sẽ đọc từ gối ba lần cho bạn nghe.
0741. Tôi đã viết từ đoạt lên bảng để bạn xem.
0742. Khi gặp từ đoạt, bạn đừng ngắt giữa chừng.
0743. Chúng ta gặp từ mương trong bài đọc hôm qua.
0744. Tôi đánh dấu từ mương bằng bút đỏ.
0745. Mọi người cùng nhìn vào từ phốn trên màn hình.
0746. Cuối buổi, tôi sẽ kiểm tra lại từ phốn.
0747. Cái từ mí này xuất hiện khá ít trong sách.
0748. Bạn thử nghe lại từ mí một lần nữa nhé.
0749. Bài học hôm nay có từ loài ở ngay đầu câu.
0750. Từ loài nằm ở giữa câu, không đọc quá nhanh.
0751. Bạn có thể đọc từ ngoan to hơn một chút.
0752. Nếu đọc sai từ ngoan thì đọc lại từ đầu.
0753. Hôm nay tôi muốn đọc từ bà thật rõ ràng.
0754. Từ bà nghe hơi lạ, nhưng rất dễ đọc.
0755. Trong câu này có từ gián, đọc chậm thôi.
0756. Tôi sẽ đọc từ gián ba lần cho bạn nghe.
0757. Tôi đã viết từ vở lên bảng để bạn xem.
0758. Khi gặp từ vở, bạn đừng ngắt giữa chừng.
0759. Chúng ta gặp từ phẫn trong bài đọc hôm qua.
0760. Tôi đánh dấu từ phẫn bằng bút đỏ.
0761. Mọi người cùng nhìn vào từ thoải trên màn hình.
0762. Cuối buổi, tôi sẽ kiểm tra lại từ thoải.
0763. Cái từ mái này xuất hiện khá ít trong sách.
0764. Bạn thử nghe lại từ mái một lần nữa nhé.
0765. Bài học hôm nay có từ vía ở ngay đầu câu.
0766. Từ vía nằm ở giữa câu, không đọc quá nhanh.
0767. Bạn có thể đọc từ vú to hơn một chút.
0768. Nếu đọc sai từ vú thì đọc lại từ đầu.
0769. Hôm nay tôi muốn đọc từ hãi thật rõ ràng.
0770. Từ hãi nghe hơi lạ, nhưng rất dễ đọc.
0771. Trong câu này có từ sắm, đọc chậm thôi.
0772. Tôi sẽ đọc từ sắm ba lần cho bạn nghe.
0773. Tôi đã viết từ lãn lên bảng để bạn xem.
0774. Khi gặp từ lãn, bạn đừng ngắt giữa chừng.
0775. Chúng ta gặp từ mặn trong bài đọc hôm qua.
0776. Tôi đánh dấu từ mặn bằng bút đỏ.
0777. Mọi người cùng nhìn vào từ lùi trên màn hình.
0778. Cuối buổi, tôi sẽ kiểm tra lại từ lùi.
0779. Cái từ phán này xuất hiện khá ít trong sách.
0780. Bạn thử nghe lại từ phán một lần nữa nhé.
0781. Bài học hôm nay có từ chủng ở ngay đầu câu.
0782. Từ chủng nằm ở giữa câu, không đọc quá nhanh.
0783. Bạn có thể đọc từ ớt to hơn một chút.
0784. Nếu đọc sai từ ớt thì đọc lại từ đầu.
0785. Hôm nay tôi muốn đọc từ hồn thật rõ ràng.
0786. Từ hồn nghe hơi lạ, nhưng rất dễ đọc.
0787. Trong câu này có từ châm, đọc chậm thôi.
0788. Tôi sẽ đọc từ châm ba lần cho bạn nghe.
0789. Tôi đã viết từ lăng lên bảng để bạn xem.
0790. Khi gặp từ lăng, bạn đừng ngắt giữa chừng.
0791. Chúng ta gặp từ giàu trong bài đọc hôm qua.
0792. Tôi đánh dấu từ giàu bằng bút đỏ.
0793. Mọi người cùng nhìn vào từ heo trên màn hình.
0794. Cuối buổi, tôi sẽ kiểm tra lại từ heo.
0795. Cái từ chó này xuất hiện khá ít trong sách.
0796. Bạn thử nghe lại từ chó một lần nữa nhé.
0797. Bài học hôm nay có từ phật ở ngay đầu câu.
0798. Từ phật nằm ở giữa câu, không đọc quá nhanh.
0799. Bạn có thể đọc từ â to hơn một chút.
0800. Nếu đọc sai từ â thì đọc lại từ đầu.
0801. Hôm nay tôi muốn đọc từ chây thật rõ ràng.
0802. Từ chây nghe hơi lạ, nhưng rất dễ đọc.
0803. Trong câu này có từ kẹo, đọc chậm thôi.
0804. Tôi sẽ đọc từ kẹo ba lần cho bạn nghe.
0805. Tôi đã viết từ thầng lên bảng để bạn xem.
0806. Khi gặp từ thầng, bạn đừng ngắt giữa chừng.
0807. Chúng ta gặp từ vại trong bài đọc hôm qua.
0808. Tôi đánh dấu từ vại bằng bút đỏ.
0809. Mọi người cùng nhìn vào từ sốt trên màn hình.
0810. Cuối buổi, tôi sẽ kiểm tra lại từ sốt.
0811. Cái từ đung này xuất hiện khá ít trong sách.
0812. Bạn thử nghe lại từ đung một lần nữa nhé.
0813. Bài học hôm nay có từ kiệt ở ngay đầu câu.
0814. Từ kiệt nằm ở giữa câu, không đọc quá nhanh.
0815. Bạn có thể đọc từ hải to hơn một chút.
0816. Nếu đọc sai từ hải thì đọc lại từ đầu.
0817. Hôm nay tôi muốn đọc từ tép thật rõ ràng.
0818. Từ tép nghe hơi lạ, nhưng rất dễ đọc.
0819. Trong câu này có từ tiệm, đọc chậm thôi.
0820. Tôi sẽ đọc từ tiệm ba lần cho bạn nghe.
0821. Tôi đã viết từ san lên bảng để bạn xem.
0822. Khi gặp từ san, bạn đừng ngắt giữa chừng.
0823. Chúng ta gặp từ ngoạch trong bài đọc hôm qua.
0824. Tôi đánh dấu từ ngoạch bằng bút đỏ.
0825. Mọi người cùng nhìn vào từ đà trên màn hình.
0826. Cuối buổi, tôi sẽ kiểm tra lại từ đà.
0827. Cái từ then này xuất hiện khá ít trong sách.
0828. Bạn thử nghe lại từ then một lần nữa nhé.
0829. Bài học hôm nay có từ thỏ ở ngay đầu câu.
0830. Từ thỏ nằm ở giữa câu, không đọc quá nhanh.
0831. Bạn có thể đọc từ thàng to hơn một chút.
0832. Nếu đọc sai từ thàng thì đọc lại từ đầu.
0833. Hôm nay tôi muốn đọc từ khít thật rõ ràng.
0834. Từ khít nghe hơi lạ, nhưng rất dễ đọc.
0835. Trong câu này có từ rỗng, đọc chậm thôi.
0836. Tôi sẽ đọc từ rỗng ba lần cho bạn nghe.
0837. Tôi đã viết từ vẫ lên bảng để bạn xem.
0838. Khi gặp từ vẫ, bạn đừng ngắt giữa chừng.
0839. Chúng ta gặp từ khây trong bài đọc hôm qua.
0840. Tôi đánh dấu từ khây bằng bút đỏ.
0841. Mọi người cùng nhìn vào từ kẹp trên màn hình.
0842. Cuối buổi, tôi sẽ kiểm tra lại từ kẹp.
0843. Cái từ phă này xuất hiện khá ít trong sách.
0844. Bạn thử nghe lại từ phă một lần nữa nhé.
0845. Bài học hôm nay có từ ghém ở ngay đầu câu.
0846. Từ ghém nằm ở giữa câu, không đọc quá nhanh.
0847. Bạn có thể đọc từ đên to hơn một chút.
0848. Nếu đọc sai từ đên thì đọc lại từ đầu.
0849. Hôm nay tôi muốn đọc từ tú thật rõ ràng.
0850. Từ tú nghe hơi lạ, nhưng rất dễ đọc.
0851. Trong câu này có từ đư, đọc chậm thôi.
0852. Tôi sẽ đọc từ đư ba lần cho bạn nghe.
0853. Tôi đã viết từ hịu lên bảng để bạn xem.
0854. Khi gặp từ hịu, bạn đừng ngắt giữa chừng.
0855. Chúng ta gặp từ dưỡng trong bài đọc hôm qua.
0856. Tôi đánh dấu từ dưỡng bằng bút đỏ.
0857. Mọi người cùng nhìn vào từ sếp trên màn hình.
0858. Cuối buổi, tôi sẽ kiểm tra lại từ sếp.
0859. Cái từ gào này xuất hiện khá ít trong sách.
0860. Bạn thử nghe lại từ gào một lần nữa nhé.
0861. Bài học hôm nay có từ ngực ở ngay đầu câu.
0862. Từ ngực nằm ở giữa câu, không đọc quá nhanh.
0863. Bạn có thể đọc từ ồi to hơn một chút.
0864. Nếu đọc sai từ ồi thì đọc lại từ đầu.
0865. Hôm nay tôi muốn đọc từ mom thật rõ ràng.
0866. Từ mom nghe hơi lạ, nhưng rất dễ đọc.
0867. Trong câu này có từ xạ, đọc chậm thôi.
0868. Tôi sẽ đọc từ xạ ba lần cho bạn nghe.
0869. Tôi đã viết từ đầm lên bảng để bạn xem.
0870. Khi gặp từ đầm, bạn đừng ngắt giữa chừng.
0871. Chúng ta gặp từ lé trong bài đọc hôm qua.
0872. Tôi đánh dấu từ lé bằng bút đỏ.
0873. Mọi người cùng nhìn vào từ thà trên màn hình.
0874. Cuối buổi, tôi sẽ kiểm tra lại từ thà.
0875. Cái từ thào này xuất hiện khá ít trong sách.
0876. Bạn thử nghe lại từ thào một lần nữa nhé.
0877. Bài học hôm nay có từ đú ở ngay đầu câu.
0878. Từ đú nằm ở giữa câu, không đọc quá nhanh.
0879. Bạn có thể đọc từ báng to hơn một chút.
0880. Nếu đọc sai từ báng thì đọc lại từ đầu.
0881. Hôm nay tôi muốn đọc từ nhão thật rõ ràng.
0882. Từ nhão nghe hơi lạ, nhưng rất dễ đọc.
0883. Trong câu này có từ chim, đọc chậm thôi.
0884. Tôi sẽ đọc từ chim ba lần cho bạn nghe.
0885. Tôi đã viết từ chảy lên bảng để bạn xem.
0886. Khi gặp từ chảy, bạn đừng ngắt giữa chừng.
0887. Chúng ta gặp từ ră trong bài đọc hôm qua.
0888. Tôi đánh dấu từ ră bằng bút đỏ.
0889. Mọi người cùng nhìn vào từ nhẹo trên màn hình.
0890. Cuối buổi, tôi sẽ kiểm tra lại từ nhẹo.
0891. Cái từ ái này xuất hiện khá ít trong sách.
0892. Bạn thử nghe lại từ ái một lần nữa nhé.
0893. Bài học hôm nay có từ hoạc ở ngay đầu câu.
0894. Từ hoạc nằm ở giữa câu, không đọc quá nhanh.
0895. Bạn có thể đọc từ việu to hơn một chút.
0896. Nếu đọc sai từ việu thì đọc lại từ đầu.
0897. Hôm nay tôi muốn đọc từ chê thật rõ ràng.
0898. Từ chê nghe hơi lạ, nhưng rất dễ đọc.
0899. Trong câu này có từ ngọng, đọc chậm thôi.
0900. Tôi sẽ đọc từ ngọng ba lần cho bạn nghe.
0901. Tôi đã viết từ nằng lên bảng để bạn xem.
0902. Khi gặp từ nằng, bạn đừng ngắt giữa chừng.
0903. Chúng ta gặp từ xửu trong bài đọc hôm qua.
0904. Tôi đánh dấu từ xửu bằng bút đỏ.
0905. Mọi người cùng nhìn vào từ sóng trên màn hình.
0906. Cuối buổi, tôi sẽ kiểm tra lại từ sóng.
0907. Cái từ cỏ này xuất hiện khá ít trong sách.
0908. Bạn thử nghe lại từ cỏ một lần nữa nhé.
0909. Bài học hôm nay có từ chênh ở ngay đầu câu.
0910. Từ chênh nằm ở giữa câu, không đọc quá nhanh.
0911. Bạn có thể đọc từ cạch to hơn một chút.
0912. Nếu đọc sai từ cạch thì đọc lại từ đầu.
0913. Hôm nay tôi muốn đọc từ tiễn thật rõ ràng.
0914. Từ tiễn nghe hơi lạ, nhưng rất dễ đọc.
0915. Trong câu này có từ lạch, đọc chậm thôi.
0916. Tôi sẽ đọc từ lạch ba lần cho bạn nghe.
0917. Tôi đã viết từ lớc lên bảng để bạn xem.
0918. Khi gặp từ lớc, bạn đừng ngắt giữa chừng.
0919. Chúng ta gặp từ nấc trong bài đọc hôm qua.
0920. Tôi đánh dấu từ nấc bằng bút đỏ.
0921. Mọi người cùng nhìn vào từ thệ trên màn hình.
0922. Cuối buổi, tôi sẽ kiểm tra lại từ thệ.
0923. Cái từ thủa này xuất hiện khá ít trong sách.
0924. Bạn thử nghe lại từ thủa một lần nữa nhé.
0925. Bài học hôm nay có từ quà ở ngay đầu câu.
0926. Từ quà nằm ở giữa câu, không đọc quá nhanh.
0927. Bạn có thể đọc từ cựu to hơn một chút.
0928. Nếu đọc sai từ cựu thì đọc lại từ đầu.
0929. Hôm nay tôi muốn đọc từ nặp thật rõ ràng.
0930. Từ nặp nghe hơi lạ, nhưng rất dễ đọc.
0931. Trong câu này có từ buồng, đọc chậm thôi.
0932. Tôi sẽ đọc từ buồng ba lần cho bạn nghe.
0933. Tôi đã viết từ vắng lên bảng để bạn xem.
0934. Khi gặp từ vắng, bạn đừng ngắt giữa chừng.
0935. Chúng ta gặp từ thầ trong bài đọc hôm qua.
0936. Tôi đánh dấu từ thầ bằng bút đỏ.
0937. Mọi người cùng nhìn vào từ rá trên màn hình.
0938. Cuối buổi, tôi sẽ kiểm tra lại từ rá.
0939. Cái từ nhưu này xuất hiện khá ít trong sách.
0940. Bạn thử nghe lại từ nhưu một lần nữa nhé.
0941. Bài học hôm nay có từ hổng ở ngay đầu câu.
0942. Từ hổng nằm ở giữa câu, không đọc quá nhanh.
0943. Bạn có thể đọc từ giào to hơn một chút.
0944. Nếu đọc sai từ giào thì đọc lại từ đầu.
0945. Hôm nay tôi muốn đọc từ cọc thật rõ ràng.
0946. Từ cọc nghe hơi lạ, nhưng rất dễ đọc.
0947. Trong câu này có từ quấy, đọc chậm thôi.
0948. Tôi sẽ đọc từ quấy ba lần cho bạn nghe.
0949. Tôi đã viết từ lường lên bảng để bạn xem.
0950. Khi gặp từ lường, bạn đừng ngắt giữa chừng.
0951. Chúng ta gặp từ giái trong bài đọc hôm qua.
0952. Tôi đánh dấu từ giái bằng bút đỏ.
0953. Mọi người cùng nhìn vào từ sớ trên màn hình.
0954. Cuối buổi, tôi sẽ kiểm tra lại từ sớ.
0955. Cái từ họng này xuất hiện khá ít trong sách.
0956. Bạn thử nghe lại từ họng một lần nữa nhé.
0957. Bài học hôm nay có từ nịt ở ngay đầu câu.
0958. Từ nịt nằm ở giữa câu, không đọc quá nhanh.
0959. Bạn có thể đọc từ vệp to hơn một chút.
0960. Nếu đọc sai từ vệp thì đọc lại từ đầu.
0961. Hôm nay tôi muốn đọc từ phởi thật rõ ràng.
0962. Từ phởi nghe hơi lạ, nhưng rất dễ đọc.
0963. Trong câu này có từ tanh, đọc chậm thôi.
0964. Tôi sẽ đọc từ tanh ba lần cho bạn nghe.
0965. Tôi đã viết từ ngăn lên bảng để bạn xem.
0966. Khi gặp từ ngăn, bạn đừng ngắt giữa chừng.
0967. Chúng ta gặp từ trọn trong bài đọc hôm qua.
0968. Tôi đánh dấu từ trọn bằng bút đỏ.
0969. Mọi người cùng nhìn vào từ tấy trên màn hình.
0970. Cuối buổi, tôi sẽ kiểm tra lại từ tấy.
0971. Cái từ tấm này xuất hiện khá ít trong sách.
0972. Bạn thử nghe lại từ tấm một lần nữa nhé.
0973. Bài học hôm nay có từ xóm ở ngay đầu câu.
0974. Từ xóm nằm ở giữa câu, không đọc quá nhanh.
0975. Bạn có thể đọc từ chẽ to hơn một chút.
0976. Nếu đọc sai từ chẽ thì đọc lại từ đầu.
0977. Hôm nay tôi muốn đọc từ hừng thật rõ ràng.
0978. Từ hừng nghe hơi lạ, nhưng rất dễ đọc.
0979. Trong câu này có từ sủa, đọc chậm thôi.
0980. Tôi sẽ đọc từ sủa ba lần cho bạn nghe.
0981. Tôi đã viết từ hất lên bảng để bạn xem.
0982. Khi gặp từ hất, bạn đừng ngắt giữa chừng.
0983. Chúng ta gặp từ tịnh trong bài đọc hôm qua.
0984. Tôi đánh dấu từ tịnh bằng bút đỏ.
0985. Mọi người cùng nhìn vào từ dàn trên màn hình.
0986. Cuối buổi, tôi sẽ kiểm tra lại từ dàn.
0987. Cái từ khoan này xuất hiện khá ít trong sách.
0988. Bạn thử nghe lại từ khoan một lần nữa nhé.
0989. Bài học hôm nay có từ mại ở ngay đầu câu.
0990. Từ mại nằm ở giữa câu, không đọc quá nhanh.
0991. Bạn có thể đọc từ nép to hơn một chút.
0992. Nếu đọc sai từ nép thì đọc lại từ đầu.
0993. Hôm nay tôi muốn đọc từ mão thật rõ ràng.
0994. Từ mão nghe hơi lạ, nhưng rất dễ đọc.
0995. Trong câu này có từ đù, đọc chậm thôi.
0996. Tôi sẽ đọc từ đù ba lần cho bạn nghe.
0997. Tôi đã viết từ hông lên bảng để bạn xem.
0998. Khi gặp từ hông, bạn đừng ngắt giữa chừng.
0999. Chúng ta gặp từ phiếu trong bài đọc hôm qua.
1000. Tôi đánh dấu từ phiếu bằng bút đỏ.
1001. Mọi người cùng nhìn vào từ đuôi trên màn hình.
1002. Cuối buổi, tôi sẽ kiểm tra lại từ đuôi.
1003. Cái từ thước này xuất hiện khá ít trong sách.
1004. Bạn thử nghe lại từ thước một lần nữa nhé.
1005. Bài học hôm nay có từ nhộn ở ngay đầu câu.
1006. Từ nhộn nằm ở giữa câu, không đọc quá nhanh.
1007. Bạn có thể đọc từ chiết to hơn một chút.
1008. Nếu đọc sai từ chiết thì đọc lại từ đầu.
1009. Hôm nay tôi muốn đọc từ nhẫn thật rõ ràng.
1010. Từ nhẫn nghe hơi lạ, nhưng rất dễ đọc.
1011. Trong câu này có từ kháp, đọc chậm thôi.
1012. Tôi sẽ đọc từ kháp ba lần cho bạn nghe.
1013. Tôi đã viết từ vợ lên bảng để bạn xem.
1014. Khi gặp từ vợ, bạn đừng ngắt giữa chừng.
1015. Chúng ta gặp từ ướm trong bài đọc hôm qua.
1016. Tôi đánh dấu từ ướm bằng bút đỏ.
1017. Mọi người cùng nhìn vào từ khiếu trên màn hình.
1018. Cuối buổi, tôi sẽ kiểm tra lại từ khiếu.
1019. Cái từ khoái này xuất hiện khá ít trong sách.
1020. Bạn thử nghe lại từ khoái một lần nữa nhé.
1021. Bài học hôm nay có từ khói ở ngay đầu câu.
1022. Từ khói nằm ở giữa câu, không đọc quá nhanh.
1023. Bạn có thể đọc từ đặn to hơn một chút.
1024. Nếu đọc sai từ đặn thì đọc lại từ đầu.
1025. Hôm nay tôi muốn đọc từ cên thật rõ ràng.
1026. Từ cên nghe hơi lạ, nhưng rất dễ đọc.
1027. Trong câu này có từ phếp, đọc chậm thôi.
1028. Tôi sẽ đọc từ phếp ba lần cho bạn nghe.
1029. Tôi đã viết từ lang lên bảng để bạn xem.
1030. Khi gặp từ lang, bạn đừng ngắt giữa chừng.
1031. Chúng ta gặp từ gỗ trong bài đọc hôm qua.
1032. Tôi đánh dấu từ gỗ bằng bút đỏ.
1033. Mọi người cùng nhìn vào từ trụng trên màn hình.
1034. Cuối buổi, tôi sẽ kiểm tra lại từ trụng.
1035. Cái từ xám này xuất hiện khá ít trong sách.
1036. Bạn thử nghe lại từ xám một lần nữa nhé.
1037. Bài học hôm nay có từ kẹ ở ngay đầu câu.
1038. Từ kẹ nằm ở giữa câu, không đọc quá nhanh.
1039. Bạn có thể đọc từ nhả to hơn một chút.
1040. Nếu đọc sai từ nhả thì đọc lại từ đầu.
1041. Hôm nay tôi muốn đọc từ quần thật rõ ràng.
1042. Từ quần nghe hơi lạ, nhưng rất dễ đọc.
1043. Trong câu này có từ thiến, đọc chậm thôi.
1044. Tôi sẽ đọc từ thiến ba lần cho bạn nghe.
1045. Tôi đã viết từ oánh lên bảng để bạn xem.
1046. Khi gặp từ oánh, bạn đừng ngắt giữa chừng.
1047. Chúng ta gặp từ cin trong bài đọc hôm qua.
1048. Tôi đánh dấu từ cin bằng bút đỏ.
1049. Mọi người cùng nhìn vào từ tháo trên màn hình.
1050. Cuối buổi, tôi sẽ kiểm tra lại từ tháo.
1051. Cái từ đáy này xuất hiện khá ít trong sách.
1052. Bạn thử nghe lại từ đáy một lần nữa nhé.
1053. Bài học hôm nay có từ khếp ở ngay đầu câu.
1054. Từ khếp nằm ở giữa câu, không đọc quá nhanh.
1055. Bạn có thể đọc từ vau to hơn một chút.
1056. Nếu đọc sai từ vau thì đọc lại từ đầu.
1057. Hôm nay tôi muốn đọc từ dò thật rõ ràng.
1058. Từ dò nghe hơi lạ, nhưng rất dễ đọc.
1059. Trong câu này có từ xư, đọc chậm thôi.
1060. Tôi sẽ đọc từ xư ba lần cho bạn nghe.
1061. Tôi đã viết từ hui lên bảng để bạn xem.
1062. Khi gặp từ hui, bạn đừng ngắt giữa chừng.
1063. Chúng ta gặp từ cỡ trong bài đọc hôm qua.
1064. Tôi đánh dấu từ cỡ bằng bút đỏ.
1065. Mọi người cùng nhìn vào từ bàm trên màn hình.
1066. Cuối buổi, tôi sẽ kiểm tra lại từ bàm.
1067. Cái từ đé này xuất hiện khá ít trong sách.
1068. Bạn thử nghe lại từ đé một lần nữa nhé.
1069. Bài học hôm nay có từ rei ở ngay đầu câu.
1070. Từ rei nằm ở giữa câu, không đọc quá nhanh.
1071. Bạn có thể đọc từ xời to hơn một chút.
1072. Nếu đọc sai từ xời thì đọc lại từ đầu.
1073. Hôm nay tôi muốn đọc từ rey thật rõ ràng.
1074. Từ rey nghe hơi lạ, nhưng rất dễ đọc.
1075. Trong câu này có từ lau, đọc chậm thôi.
1076. Tôi sẽ đọc từ lau ba lần cho bạn nghe.
1077. Tôi đã viết từ xẻ lên bảng để bạn xem.
1078. Khi gặp từ xẻ, bạn đừng ngắt giữa chừng.
1079. Chúng ta gặp từ bồng trong bài đọc hôm qua.
1080. Tôi đánh dấu từ bồng bằng bút đỏ.
1081. Mọi người cùng nhìn vào từ bềnh trên màn hình.
1082. Cuối buổi, tôi sẽ kiểm tra lại từ bềnh.
1083. Cái từ ngẩng này xuất hiện khá ít trong sách.
1084. Bạn thử nghe lại từ ngẩng một lần nữa nhé.
1085. Bài học hôm nay có từ ghê ở ngay đầu câu.
1086. Từ ghê nằm ở giữa câu, không đọc quá nhanh.
1087. Bạn có thể đọc từ pí to hơn một chút.
1088. Nếu đọc sai từ pí thì đọc lại từ đầu.
1089. Hôm nay tôi muốn đọc từ tran thật rõ ràng.
1090. Từ tran nghe hơi lạ, nhưng rất dễ đọc.
1091. Trong câu này có từ bãy, đọc chậm thôi.
1092. Tôi sẽ đọc từ bãy ba lần cho bạn nghe.
1093. Tôi đã viết từ lỡ lên bảng để bạn xem.
1094. Khi gặp từ lỡ, bạn đừng ngắt giữa chừng.
1095. Chúng ta gặp từ khoai trong bài đọc hôm qua.
1096. Tôi đánh dấu từ khoai bằng bút đỏ.
1097. Mọi người cùng nhìn vào từ choáng trên màn hình.
1098. Cuối buổi, tôi sẽ kiểm tra lại từ choáng.
1099. Cái từ ngợp này xuất hiện khá ít trong sách.
1100. Bạn thử nghe lại từ ngợp một lần nữa nhé.
1101. Bài học hôm nay có từ điển ở ngay đầu câu.
1102. Từ điển nằm ở giữa câu, không đọc quá nhanh.
1103. Bạn có thể đọc từ đệ to hơn một chút.
1104. Nếu đọc sai từ đệ thì đọc lại từ đầu.
1105. Hôm nay tôi muốn đọc từ chúc thật rõ ràng.
1106. Từ chúc nghe hơi lạ, nhưng rất dễ đọc.
1107. Trong câu này có từ dong, đọc chậm thôi.
1108. Tôi sẽ đọc từ dong ba lần cho bạn nghe.
1109. Tôi đã viết từ mú lên bảng để bạn xem.
1110. Khi gặp từ mú, bạn đừng ngắt giữa chừng.
1111. Chúng ta gặp từ mút trong bài đọc hôm qua.
1112. Tôi đánh dấu từ mút bằng bút đỏ.
1113. Mọi người cùng nhìn vào từ dận trên màn hình.
1114. Cuối buổi, tôi sẽ kiểm tra lại từ dận.
1115. Cái từ bổi này xuất hiện khá ít trong sách.
1116. Bạn thử nghe lại từ bổi một lần nữa nhé.
1117. Bài học hôm nay có từ bôi ở ngay đầu câu.
1118. Từ bôi nằm ở giữa câu, không đọc quá nhanh.
1119. Bạn có thể đọc từ cưỡi to hơn một chút.
1120. Nếu đọc sai từ cưỡi thì đọc lại từ đầu.
1121. Hôm nay tôi muốn đọc từ thẩm thật rõ ràng.
1122. Từ thẩm nghe hơi lạ, nhưng rất dễ đọc.
1123. Trong câu này có từ toà, đọc chậm thôi.
1124. Tôi sẽ đọc từ toà ba lần cho bạn nghe.
1125. Tôi đã viết từ ngỏi lên bảng để bạn xem.
1126. Khi gặp từ ngỏi, bạn đừng ngắt giữa chừng.
1127. Chúng ta gặp từ hảo trong bài đọc hôm qua.
1128. Tôi đánh dấu từ hảo bằng bút đỏ.
1129. Mọi người cùng nhìn vào từ lình trên màn hình.
1130. Cuối buổi, tôi sẽ kiểm tra lại từ lình.
1131. Cái từ nghẽ này xuất hiện khá ít trong sách.
1132. Bạn thử nghe lại từ nghẽ một lần nữa nhé.
1133. Bài học hôm nay có từ nhọ ở ngay đầu câu.
1134. Từ nhọ nằm ở giữa câu, không đọc quá nhanh.
1135. Bạn có thể đọc từ gắng to hơn một chút.
1136. Nếu đọc sai từ gắng thì đọc lại từ đầu.
1137. Hôm nay tôi muốn đọc từ nhị thật rõ ràng.
1138. Từ nhị nghe hơi lạ, nhưng rất dễ đọc.
1139. Trong câu này có từ toả, đọc chậm thôi.
1140. Tôi sẽ đọc từ toả ba lần cho bạn nghe.
1141. Tôi đã viết từ táo lên bảng để bạn xem.
1142. Khi gặp từ táo, bạn đừng ngắt giữa chừng.
1143. Chúng ta gặp từ mũ trong bài đọc hôm qua.
1144. Tôi đánh dấu từ mũ bằng bút đỏ.
1145. Mọi người cùng nhìn vào từ xuyến trên màn hình.
1146. Cuối buổi, tôi sẽ kiểm tra lại từ xuyến.
1147. Cái từ tứ này xuất hiện khá ít trong sách.
1148. Bạn thử nghe lại từ tứ một lần nữa nhé.
1149. Bài học hôm nay có từ kì ở ngay đầu câu.
1150. Từ kì nằm ở giữa câu, không đọc quá nhanh.
1151. Bạn có thể đọc từ ngòm to hơn một chút.
1152. Nếu đọc sai từ ngòm thì đọc lại từ đầu.
1153. Hôm nay tôi muốn đọc từ trương thật rõ ràng.
1154. Từ trương nghe hơi lạ, nhưng rất dễ đọc.
1155. Trong câu này có từ bạt, đọc chậm thôi.
1156. Tôi sẽ đọc từ bạt ba lần cho bạn nghe.
1157. Tôi đã viết từ pạt lên bảng để bạn xem.
1158. Khi gặp từ pạt, bạn đừng ngắt giữa chừng.
1159. Chúng ta gặp từ ngoằng trong bài đọc hôm qua.
1160. Tôi đánh dấu từ ngoằng bằng bút đỏ.
1161. Mọi người cùng nhìn vào từ mù trên màn hình.
1162. Cuối buổi, tôi sẽ kiểm tra lại từ mù.
1163. Cái từ hoành này xuất hiện khá ít trong sách.
1164. Bạn thử nghe lại từ hoành một lần nữa nhé.
1165. Bài học hôm nay có từ oi ở ngay đầu câu.
1166. Từ oi nằm ở giữa câu, không đọc quá nhanh.
1167. Bạn có thể đọc từ bạc to hơn một chút.
1168. Nếu đọc sai từ bạc thì đọc lại từ đầu.
1169. Hôm nay tôi muốn đọc từ ưng thật rõ ràng.
1170. Từ ưng nghe hơi lạ, nhưng rất dễ đọc.
1171. Trong câu này có từ phim, đọc chậm thôi.
1172. Tôi sẽ đọc từ phim ba lần cho bạn nghe.
1173. Tôi đã viết từ răm lên bảng để bạn xem.
1174. Khi gặp từ răm, bạn đừng ngắt giữa chừng.
1175. Chúng ta gặp từ trôi trong bài đọc hôm qua.
1176. Tôi đánh dấu từ trôi bằng bút đỏ.
1177. Mọi người cùng nhìn vào từ vệ trên màn hình.
1178. Cuối buổi, tôi sẽ kiểm tra lại từ vệ.
1179. Cái từ nhện này xuất hiện khá ít trong sách.
1180. Bạn thử nghe lại từ nhện một lần nữa nhé.
1181. Bài học hôm nay có từ bừa ở ngay đầu câu.
1182. Từ bừa nằm ở giữa câu, không đọc quá nhanh.
1183. Bạn có thể đọc từ hụt to hơn một chút.
1184. Nếu đọc sai từ hụt thì đọc lại từ đầu.
1185. Hôm nay tôi muốn đọc từ chăm thật rõ ràng.
1186. Từ chăm nghe hơi lạ, nhưng rất dễ đọc.
1187. Trong câu này có từ hẹp, đọc chậm thôi.
1188. Tôi sẽ đọc từ hẹp ba lần cho bạn nghe.
1189. Tôi đã viết từ lọt lên bảng để bạn xem.
1190. Khi gặp từ lọt, bạn đừng ngắt giữa chừng.
1191. Chúng ta gặp từ đố trong bài đọc hôm qua.
1192. Tôi đánh dấu từ đố bằng bút đỏ.
1193. Mọi người cùng nhìn vào từ bức trên màn hình.
1194. Cuối buổi, tôi sẽ kiểm tra lại từ bức.
1195. Cái từ yên này xuất hiện khá ít trong sách.
1196. Bạn thử nghe lại từ yên một lần nữa nhé.
1197. Bài học hôm nay có từ ngại ở ngay đầu câu.
1198. Từ ngại nằm ở giữa câu, không đọc quá nhanh.
1199. Bạn có thể đọc từ hoi to hơn một chút.
1200. Nếu đọc sai từ hoi thì đọc lại từ đầu.
1201. Ba từ cần đọc chậm trong câu này là tở, xả và ghim.
1202. Ba từ cần đọc chậm trong câu này là lẩy, quân và neo.
1203. Ba từ cần đọc chậm trong câu này là chước, nghiêng và râu.
1204. Ba từ cần đọc chậm trong câu này là seo, gút và phác.
1205. Ba từ cần đọc chậm trong câu này là mư, chướng và thừ.
1206. Ba từ cần đọc chậm trong câu này là chử, hán và hận.
1207. Ba từ cần đọc chậm trong câu này là gặ, vặn và cò.
1208. Ba từ cần đọc chậm trong câu này là ngâm, xổ và xùa.
1209. Ba từ cần đọc chậm trong câu này là sườn, tẩy và pên.
1210. Ba từ cần đọc chậm trong câu này là điêu, cở và địp.
1211. Ba từ cần đọc chậm trong câu này là hòn, hô và khỉ.
1212. Ba từ cần đọc chậm trong câu này là trú, chập và trờn.
1213. Ba từ cần đọc chậm trong câu này là ýnh, peo và đền.
1214. Ba từ cần đọc chậm trong câu này là lậu, rèn và bướng.
1215. Ba từ cần đọc chậm trong câu này là gậy, reo và chỉa.
1216. Ba từ cần đọc chậm trong câu này là bơn, bếch và chặt.
1217. Ba từ cần đọc chậm trong câu này là cing, nướt và dượt.
1218. Ba từ cần đọc chậm trong câu này là đeo, bì và víp.
1219. Ba từ cần đọc chậm trong câu này là toan, toanh và tua.
1220. Ba từ cần đọc chậm trong câu này là chẵn, gáy và cầ.
1221. Ba từ cần đọc chậm trong câu này là đùng, lòi và lếc.
1222. Ba từ cần đọc chậm trong câu này là xộp, lạn và uầ.
1223. Ba từ cần đọc chậm trong câu này là dam, nủng và vong.
1224. Ba từ cần đọc chậm trong câu này là khoả, củi và đụng.
1225. Ba từ cần đọc chậm trong câu này là xóp, thuận và dốt.
1226. Ba từ cần đọc chậm trong câu này là đẻ, mịt và kệp.
1227. Ba từ cần đọc chậm trong câu này là móng, dăm và thoả.
1228. Ba từ cần đọc chậm trong câu này là mãn, đoản và ngoái.
1229. Ba từ cần đọc chậm trong câu này là lông, hiệp và loãng.
1230. Ba từ cần đọc chậm trong câu này là đoan, đùm và tông.
1231. Ba từ cần đọc chậm trong câu này là kiếp, thạch và quỷ.
1232. Ba từ cần đọc chậm trong câu này là nhãn, hi và dou.
1233. Ba từ cần đọc chậm trong câu này là nướn, tỉnh và kỷ.
1234. Ba từ cần đọc chậm trong câu này là loa, đau và tráng.
1235. Ba từ cần đọc chậm trong câu này là sói, diệu và thấu.
1236. Ba từ cần đọc chậm trong câu này là thiếc, nều và thịch.
1237. Ba từ cần đọc chậm trong câu này là dợ, vàn và ổ.
1238. Ba từ cần đọc chậm trong câu này là điềm, tộn và gý.
1239. Ba từ cần đọc chậm trong câu này là giặt, nhiệu và sing.
1240. Ba từ cần đọc chậm trong câu này là nang, máng và hập.
1241. Ba từ cần đọc chậm trong câu này là đói, điệp và dõ.
1242. Ba từ cần đọc chậm trong câu này là dí, treo và đúc.
1243. Ba từ cần đọc chậm trong câu này là buông, lỏng và hỏng.
1244. Ba từ cần đọc chậm trong câu này là ngỡ, ngẩn và tràn.
1245. Ba từ cần đọc chậm trong câu này là nan, ngoạn và chuộng.
1246. Ba từ cần đọc chậm trong câu này là hổ, nhằm và mâu.
1247. Ba từ cần đọc chậm trong câu này là thuẫn, rắc và rớ.
1248. Ba từ cần đọc chậm trong câu này là ngình, mẻ và dọm.
1249. Ba từ cần đọc chậm trong câu này là kụ, bão và thụt.
1250. Ba từ cần đọc chậm trong câu này là nhì, héo và tưới.
1251. Ba từ cần đọc chậm trong câu này là phóc, chàm và đỉnh.
1252. Ba từ cần đọc chậm trong câu này là núp, trự và sanh.
1253. Ba từ cần đọc chậm trong câu này là mỉ, điên và hừm.
1254. Ba từ cần đọc chậm trong câu này là xỉ, trừu và trưng.
1255. Ba từ cần đọc chậm trong câu này là nấu, răng và cưa.
1256. Ba từ cần đọc chậm trong câu này là loạng, đầ và bày.
1257. Ba từ cần đọc chậm trong câu này là nế, dệnh và lớt.
1258. Ba từ cần đọc chậm trong câu này là chăn, dục và ngã.
1259. Ba từ cần đọc chậm trong câu này là ngoạt, suốt và rỡ.
1260. Ba từ cần đọc chậm trong câu này là phươ, dẽ và thổi.
1261. Ba từ cần đọc chậm trong câu này là thượng, nộn và gắm.
1262. Ba từ cần đọc chậm trong câu này là bến, phảm và chù.
1263. Ba từ cần đọc chậm trong câu này là ngyễn, ngyện và hỏ.
1264. Ba từ cần đọc chậm trong câu này là trân, lãng và băn.
1265. Ba từ cần đọc chậm trong câu này là ủi, mưa và gió.
1266. Ba từ cần đọc chậm trong câu này là đành, phận và chạt.
1267. Ba từ cần đọc chậm trong câu này là khỏ, phé và kện.
1268. Ba từ cần đọc chậm trong câu này là ngộp, ben và sấy.
1269. Ba từ cần đọc chậm trong câu này là đôn, ming và cút.
1270. Ba từ cần đọc chậm trong câu này là hoang, guy và hời.
1271. Ba từ cần đọc chậm trong câu này là thò, kiể và tùm.
1272. Ba từ cần đọc chậm trong câu này là hí, què và thằ.
1273. Ba từ cần đọc chậm trong câu này là cuồng, vịt và còng.
1274. Ba từ cần đọc chậm trong câu này là théo, soi và ping.
1275. Ba từ cần đọc chậm trong câu này là uống, duyệu và phia.
1276. Ba từ cần đọc chậm trong câu này là săn, thuếc và giỏ.
1277. Ba từ cần đọc chậm trong câu này là toen, hu và ang.
1278. Ba từ cần đọc chậm trong câu này là nguyền, phơ và lam.
1279. Ba từ cần đọc chậm trong câu này là rát, phà và chio.
1280. Ba từ cần đọc chậm trong câu này là thươ, gạo và giục.
1281. Ba từ cần đọc chậm trong câu này là lận, miên và chợ.
1282. Ba từ cần đọc chậm trong câu này là bền, xứ và dịp.
1283. Ba từ cần đọc chậm trong câu này là ce, vã và moo.
1284. Ba từ cần đọc chậm trong câu này là lõi, vướng và phủ.
1285. Ba từ cần đọc chậm trong câu này là tiếm, huống và gắt.
1286. Ba từ cần đọc chậm trong câu này là toản, cức và bậy.
1287. Ba từ cần đọc chậm trong câu này là hút, nắp và dấy.
1288. Ba từ cần đọc chậm trong câu này là mề, đạch và nẻ.
1289. Ba từ cần đọc chậm trong câu này là nũ, nụt và oẳng.
1290. Ba từ cần đọc chậm trong câu này là thấm, khẹn và lân.
1291. Ba từ cần đọc chậm trong câu này là há, rụng và thểo.
1292. Ba từ cần đọc chậm trong câu này là nện, đở và nháp.
1293. Ba từ cần đọc chậm trong câu này là nghẽn, tuổi và nòng.
1294. Ba từ cần đọc chậm trong câu này là huyết, phấn và rầm.
1295. Ba từ cần đọc chậm trong câu này là rộ, phên và thê.
1296. Ba từ cần đọc chậm trong câu này là cấm, chệt và giành.
1297. Ba từ cần đọc chậm trong câu này là trơn, chểm và pem.
1298. Ba từ cần đọc chậm trong câu này là trẻ, nghèo và bở.
1299. Ba từ cần đọc chậm trong câu này là rủ, khắc và phẩm.
1300. Ba từ cần đọc chậm trong câu này là dâ, mồn và trài.
1301. Ba từ cần đọc chậm trong câu này là ngừng, xấu và sực.
1302. Ba từ cần đọc chậm trong câu này là bể, bơi và khét.
1303. Ba từ cần đọc chậm trong câu này là binh, nóng và nòi.
1304. Ba từ cần đọc chậm trong câu này là hú, cóp và nhần.
1305. Ba từ cần đọc chậm trong câu này là phúc, nạn và quyên.
1306. Ba từ cần đọc chậm trong câu này là đài, nhài và hiếm.
1307. Ba từ cần đọc chậm trong câu này là túp, đục và chọc.
1308. Ba từ cần đọc chậm trong câu này là đâm, hật và tậm.
1309. Ba từ cần đọc chậm trong câu này là nừa, hoặ và thoá.
1310. Ba từ cần đọc chậm trong câu này là xí, hợi và bả.
1311. Ba từ cần đọc chậm trong câu này là đing, bóp và rả.
1312. Ba từ cần đọc chậm trong câu này là gặng, só và phọc.
1313. Ba từ cần đọc chậm trong câu này là hãng, xai và nột.
1314. Ba từ cần đọc chậm trong câu này là đíp, hạ và dắm.
1315. Ba từ cần đọc chậm trong câu này là đoảng, độc và trăng.
1316. Ba từ cần đọc chậm trong câu này là hắn, trội và tuện.
1317. Ba từ cần đọc chậm trong câu này là chỗi, trễ và lạt.
1318. Ba từ cần đọc chậm trong câu này là hoán, thoạt và bẩn.
1319. Ba từ cần đọc chậm trong câu này là quăng, hỡi và nời.
1320. Ba từ cần đọc chậm trong câu này là dũng, nhường và thịnh.
1321. Ba từ cần đọc chậm trong câu này là vượng, triều và kiêu.
1322. Ba từ cần đọc chậm trong câu này là nạo, ách và ngai.
1323. Ba từ cần đọc chậm trong câu này là mạo, sò và quỳ.
1324. Ba từ cần đọc chậm trong câu này là gối, đoạt và mương.
1325. Ba từ cần đọc chậm trong câu này là phốn, mí và loài.
1326. Ba từ cần đọc chậm trong câu này là ngoan, bà và gián.
1327. Ba từ cần đọc chậm trong câu này là vở, phẫn và thoải.
1328. Ba từ cần đọc chậm trong câu này là mái, vía và vú.
1329. Ba từ cần đọc chậm trong câu này là hãi, sắm và lãn.
1330. Ba từ cần đọc chậm trong câu này là mặn, lùi và phán.
1331. Ba từ cần đọc chậm trong câu này là chủng, ớt và hồn.
1332. Ba từ cần đọc chậm trong câu này là châm, lăng và giàu.
1333. Ba từ cần đọc chậm trong câu này là heo, chó và phật.
1334. Ba từ cần đọc chậm trong câu này là â, chây và kẹo.
1335. Ba từ cần đọc chậm trong câu này là thầng, vại và sốt.
1336. Ba từ cần đọc chậm trong câu này là đung, kiệt và hải.
1337. Ba từ cần đọc chậm trong câu này là tép, tiệm và san.
1338. Ba từ cần đọc chậm trong câu này là ngoạch, đà và then.
1339. Ba từ cần đọc chậm trong câu này là thỏ, thàng và khít.
1340. Ba từ cần đọc chậm trong câu này là rỗng, vẫ và khây.
1341. Ba từ cần đọc chậm trong câu này là kẹp, phă và ghém.
1342. Ba từ cần đọc chậm trong câu này là đên, tú và đư.
1343. Ba từ cần đọc chậm trong câu này là hịu, dưỡng và sếp.
1344. Ba từ cần đọc chậm trong câu này là gào, ngực và ồi.
1345. Ba từ cần đọc chậm trong câu này là mom, xạ và đầm.
1346. Ba từ cần đọc chậm trong câu này là lé, thà và thào.
1347. Ba từ cần đọc chậm trong câu này là đú, báng và nhão.
1348. Ba từ cần đọc chậm trong câu này là chim, chảy và ră.
1349. Ba từ cần đọc chậm trong câu này là nhẹo, ái và hoạc.
1350. Ba từ cần đọc chậm trong câu này là việu, chê và ngọng.
1351. Ba từ cần đọc chậm trong câu này là nằng, xửu và sóng.
1352. Ba từ cần đọc chậm trong câu này là cỏ, chênh và cạch.
1353. Ba từ cần đọc chậm trong câu này là tiễn, lạch và lớc.
1354. Ba từ cần đọc chậm trong câu này là nấc, thệ và thủa.
1355. Ba từ cần đọc chậm trong câu này là quà, cựu và nặp.
1356. Ba từ cần đọc chậm trong câu này là buồng, vắng và thầ.
1357. Ba từ cần đọc chậm trong câu này là rá, nhưu và hổng.
1358. Ba từ cần đọc chậm trong câu này là giào, cọc và quấy.
1359. Ba từ cần đọc chậm trong câu này là lường, giái và sớ.
1360. Ba từ cần đọc chậm trong câu này là họng, nịt và vệp.
1361. Ba từ cần đọc chậm trong câu này là phởi, tanh và ngăn.
1362. Ba từ cần đọc chậm trong câu này là trọn, tấy và tấm.
1363. Ba từ cần đọc chậm trong câu này là xóm, chẽ và hừng.
1364. Ba từ cần đọc chậm trong câu này là sủa, hất và tịnh.
1365. Ba từ cần đọc chậm trong câu này là dàn, khoan và mại.
1366. Ba từ cần đọc chậm trong câu này là nép, mão và đù.
1367. Ba từ cần đọc chậm trong câu này là hông, phiếu và đuôi.
1368. Ba từ cần đọc chậm trong câu này là thước, nhộn và chiết.
1369. Ba từ cần đọc chậm trong câu này là nhẫn, kháp và vợ.
1370. Ba từ cần đọc chậm trong câu này là ướm, khiếu và khoái.
1371. Ba từ cần đọc chậm trong câu này là khói, đặn và cên.
1372. Ba từ cần đọc chậm trong câu này là phếp, lang và gỗ.
1373. Ba từ cần đọc chậm trong câu này là trụng, xám và kẹ.
1374. Ba từ cần đọc chậm trong câu này là nhả, quần và thiến.
1375. Ba từ cần đọc chậm trong câu này là oánh, cin và tháo.
1376. Ba từ cần đọc chậm trong câu này là đáy, khếp và vau.
1377. Ba từ cần đọc chậm trong câu này là dò, xư và hui.
1378. Ba từ cần đọc chậm trong câu này là cỡ, bàm và đé.
1379. Ba từ cần đọc chậm trong câu này là rei, xời và rey.
1380. Ba từ cần đọc chậm trong câu này là lau, xẻ và bồng.
1381. Ba từ cần đọc chậm trong câu này là bềnh, ngẩng và ghê.
1382. Ba từ cần đọc chậm trong câu này là pí, tran và bãy.
1383. Ba từ cần đọc chậm trong câu này là lỡ, khoai và choáng.
1384. Ba từ cần đọc chậm trong câu này là ngợp, điển và đệ.
1385. Ba từ cần đọc chậm trong câu này là chúc, dong và mú.
1386. Ba từ cần đọc chậm trong câu này là mút, dận và bổi.
1387. Ba từ cần đọc chậm trong câu này là bôi, cưỡi và thẩm.
1388. Ba từ cần đọc chậm trong câu này là toà, ngỏi và hảo.
1389. Ba từ cần đọc chậm trong câu này là lình, nghẽ và nhọ.
1390. Ba từ cần đọc chậm trong câu này là gắng, nhị và toả.
1391. Ba từ cần đọc chậm trong câu này là táo, mũ và xuyến.
1392. Ba từ cần đọc chậm trong câu này là tứ, kì và ngòm.
1393. Ba từ cần đọc chậm trong câu này là trương, bạt và pạt.
1394. Ba từ cần đọc chậm trong câu này là ngoằng, mù và hoành.
1395. Ba từ cần đọc chậm trong câu này là oi, bạc và ưng.
1396. Ba từ cần đọc chậm trong câu này là phim, răm và trôi.
1397. Ba từ cần đọc chậm trong câu này là vệ, nhện và bừa.
1398. Ba từ cần đọc chậm trong câu này là hụt, chăm và hẹp.
1399. Ba từ cần đọc chậm trong câu này là lọt, đố và bức.
1400. Ba từ cần đọc chậm trong câu này là yên, ngại và hoi.

## Câu cầu nối (đo chênh kênh thu — đọc tự nhiên như bình thường)

1401. Thì hôm nay mình sẽ hướng dẫn các bạn
1402. cao người dùng sẽ dính trên tường
1403. vẽ khối sườn trần nhà tạo một Được rồi Bây giờ
1404. Đấy Và khi đấy thì mọi
1405. Không biết nó có chưa nhưng mà mình cứ bảo thế là có rồi là nó không cần làm mà chưa có thì nó sẽ làm thêm cho mình
1406. một cái cờ re đen sần cho thêm một cái áp mới
1407. cô mít và đẩy cốt lên
1408. tiếp theo để mình cải thiện giao diện này cải thiện tính năng này
1409. Đấy đăng xuất khỏi trái đất
1410. Thế thì khả năng cao là
1411. nó chính là một cái công cụ rất là mạnh
1412. hoặc là rất nhiều các cái
1413. đến cái rờ ô nơ síp này mình cũng thấy nó khá là phức tạp
1414. để nó pho mát cho mình khá là hay
1415. cái phiên bản đấy thì nó rất là phổ biến thôi gít và gít hắp ấy
1416. Cảm ơn và mong a nhiều sự góp ý hơn Trân trọng Bên này mình thấy đấy
1417. Đây này nó đang hết luôn này
1418. Đấy nhưng mà nhưng mà cái ở đây ấy là lịch của tôi
1419. Cái côn tếch của cái cuộc trò chuyện này mình đang không biết là nó là bao nhiêu
1420. ai khiến tạo cái tài liệu thế
1421. Nhưng mà mình vào mình đọc cốt này bây giờ mình bắt đầu mình cũng thấy lú rồi
1422. tiếng Việt nên là các bạn nhìn thấy cái mật khẩu của mình Đấy
1423. rồi mình đã đăng ký thành công rồi nhá Mình sẽ su đô áp ắp đết
1424. Từ từ để đọc kỹ hướng dẫn sử dụng trước khi dùng đã
1425. thì mình có thể giả sử men tơ mình cũng sẽ tính là men tơ có thể đặt cái
1426. Đấy men tơ ở đây thì là vai trò mặc định nhá
1427. Điểm khác biệt lớn nhất ở đây là tiền từ thiện không rút tiền
1428. giảm kiểu gì ấy Thôi kệ nó đi Vậy
1429. Ở đây là đây bạn Phạm Đạt đây
1430. Thường thường là cái cấu hình này đáng ra là
1431. mình bốn trăm vào my con phích
1432. mạnh mẽ bằng cái mốt on lai như là gu gồ giê mi ni
1433. Thật ra về cách con người tê chúng ta a liệu
1434. đấy mình cũng có thể tương tác với cả ây ai nhưng mà cho ây ai lại đi tương tác với ây ai khác sau đó mình
1435. giá trị của en trô pích này qua en vê để mình đổi en vê được
1436. Ở đây mình mở lại go đót
1437. này vào Nó biết chưa biết nó viết cái gì mình viết nguyên vào đã
1438. là em với cả lây ơ một Thì cái thằng cái thằng một này
1439. da nhưng mà phông chữ có vẻ có bé nhỉ
1440. mình viết luôn ưu điểm vào trong cái cái này đi
1441. Nhưng mà huỷ ở đây ấ thì
1442. Mình nghĩ là cái phần đánh giá
1443. Đây nhá Đây là phân tích nghiệp vụ này
1444. Sau đó mình sẽ tạo một cái mét in sờ tần bo đi Đó cái mét này khả năng cao là để vẽ ra cái mặt đất đấy
1445. nếu mà cái hàm mà i ét on phờ lo của cái thằng ca rác tơ bo đi ba dê này mà tru ấy
1446. Trong in xờ pếch tơ chọn mơ ti ri ồ xờ tan đơ
1447. Thì ở trong mên này thì sẽ tạo luôn một cái nút côn trôn
1448. tạo một cái chung sau đó là tạo cho cái thằng a ni mê sần của cái thằng mốp cũng giống à
1449. ở cháp tơ một thì chúng ta sẽ thấy cái điểm chính của chương trình sẽ là
1450. thứ nhất là phẩy thì là đối số thứ nhất

## Cặp tối thiểu (phân biệt phụ âm cuối / vần)

1451. Tôi đọc cặp từ này: xanh, rồi xan. Nghe lại lần nữa: xanh, xan.
1452. Tôi đọc cặp từ này: ách, rồi át. Nghe lại lần nữa: ách, át.
1453. Tôi đọc cặp từ này: ếch, rồi ết. Nghe lại lần nữa: ếch, ết.
1454. Tôi đọc cặp từ này: lạnh, rồi lạn. Nghe lại lần nữa: lạnh, lạn.
1455. Tôi đọc cặp từ này: quanh, rồi quan. Nghe lại lần nữa: quanh, quan.
1456. Tôi đọc cặp từ này: kênh, rồi kên. Nghe lại lần nữa: kênh, kên.
1457. Tôi đọc cặp từ này: minh, rồi min. Nghe lại lần nữa: minh, min.
1458. Tôi đọc cặp từ này: móc, rồi mót. Nghe lại lần nữa: móc, mót.
1459. Tôi đọc cặp từ này: cốc, rồi cốt. Nghe lại lần nữa: cốc, cốt.
1460. Tôi đọc cặp từ này: múc, rồi mút. Nghe lại lần nữa: múc, mút.
1461. Tôi đọc cặp từ này: kẹo, rồi kẻo. Nghe lại lần nữa: kẹo, kẻo.
1462. Tôi đọc cặp từ này: boong, rồi bong. Nghe lại lần nữa: boong, bong.
1463. Tôi đọc cặp từ này: khuých, rồi khuých. Nghe lại lần nữa: khuých, khuých.
1464. Tôi đọc cặp từ này: oách, rồi oát. Nghe lại lần nữa: oách, oát.
1465. Tôi đọc cặp từ này: xanh, rồi xan. Nghe lại lần nữa: xanh, xan.
1466. Tôi đọc cặp từ này: ách, rồi át. Nghe lại lần nữa: ách, át.
1467. Tôi đọc cặp từ này: ếch, rồi ết. Nghe lại lần nữa: ếch, ết.
1468. Tôi đọc cặp từ này: lạnh, rồi lạn. Nghe lại lần nữa: lạnh, lạn.
1469. Tôi đọc cặp từ này: quanh, rồi quan. Nghe lại lần nữa: quanh, quan.
1470. Tôi đọc cặp từ này: kênh, rồi kên. Nghe lại lần nữa: kênh, kên.
1471. Tôi đọc cặp từ này: minh, rồi min. Nghe lại lần nữa: minh, min.
1472. Tôi đọc cặp từ này: móc, rồi mót. Nghe lại lần nữa: móc, mót.
1473. Tôi đọc cặp từ này: cốc, rồi cốt. Nghe lại lần nữa: cốc, cốt.
1474. Tôi đọc cặp từ này: múc, rồi mút. Nghe lại lần nữa: múc, mút.
1475. Tôi đọc cặp từ này: kẹo, rồi kẻo. Nghe lại lần nữa: kẹo, kẻo.
1476. Tôi đọc cặp từ này: boong, rồi bong. Nghe lại lần nữa: boong, bong.
1477. Tôi đọc cặp từ này: khuých, rồi khuých. Nghe lại lần nữa: khuých, khuých.
1478. Tôi đọc cặp từ này: oách, rồi oát. Nghe lại lần nữa: oách, oát.
1479. Tôi đọc cặp từ này: xanh, rồi xan. Nghe lại lần nữa: xanh, xan.
1480. Tôi đọc cặp từ này: ách, rồi át. Nghe lại lần nữa: ách, át.
1481. Tôi đọc cặp từ này: ếch, rồi ết. Nghe lại lần nữa: ếch, ết.
1482. Tôi đọc cặp từ này: lạnh, rồi lạn. Nghe lại lần nữa: lạnh, lạn.
1483. Tôi đọc cặp từ này: quanh, rồi quan. Nghe lại lần nữa: quanh, quan.
1484. Tôi đọc cặp từ này: kênh, rồi kên. Nghe lại lần nữa: kênh, kên.
1485. Tôi đọc cặp từ này: minh, rồi min. Nghe lại lần nữa: minh, min.
1486. Tôi đọc cặp từ này: móc, rồi mót. Nghe lại lần nữa: móc, mót.
1487. Tôi đọc cặp từ này: cốc, rồi cốt. Nghe lại lần nữa: cốc, cốt.
1488. Tôi đọc cặp từ này: múc, rồi mút. Nghe lại lần nữa: múc, mút.
1489. Tôi đọc cặp từ này: kẹo, rồi kẻo. Nghe lại lần nữa: kẹo, kẻo.
1490. Tôi đọc cặp từ này: boong, rồi bong. Nghe lại lần nữa: boong, bong.
1491. Tôi đọc cặp từ này: khuých, rồi khuých. Nghe lại lần nữa: khuých, khuých.
1492. Tôi đọc cặp từ này: oách, rồi oát. Nghe lại lần nữa: oách, oát.
1493. Tôi đọc cặp từ này: xanh, rồi xan. Nghe lại lần nữa: xanh, xan.
1494. Tôi đọc cặp từ này: ách, rồi át. Nghe lại lần nữa: ách, át.
1495. Tôi đọc cặp từ này: ếch, rồi ết. Nghe lại lần nữa: ếch, ết.
1496. Tôi đọc cặp từ này: lạnh, rồi lạn. Nghe lại lần nữa: lạnh, lạn.
1497. Tôi đọc cặp từ này: quanh, rồi quan. Nghe lại lần nữa: quanh, quan.
1498. Tôi đọc cặp từ này: kênh, rồi kên. Nghe lại lần nữa: kênh, kên.
1499. Tôi đọc cặp từ này: minh, rồi min. Nghe lại lần nữa: minh, min.
1500. Tôi đọc cặp từ này: móc, rồi mót. Nghe lại lần nữa: móc, mót.
1501. Tôi đọc cặp từ này: cốc, rồi cốt. Nghe lại lần nữa: cốc, cốt.
1502. Tôi đọc cặp từ này: múc, rồi mút. Nghe lại lần nữa: múc, mút.
1503. Tôi đọc cặp từ này: kẹo, rồi kẻo. Nghe lại lần nữa: kẹo, kẻo.
1504. Tôi đọc cặp từ này: boong, rồi bong. Nghe lại lần nữa: boong, bong.
1505. Tôi đọc cặp từ này: khuých, rồi khuých. Nghe lại lần nữa: khuých, khuých.
1506. Tôi đọc cặp từ này: oách, rồi oát. Nghe lại lần nữa: oách, oát.
