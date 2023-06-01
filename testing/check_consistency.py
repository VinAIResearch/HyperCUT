import torch
from testing.tester import Tester


class ConsistencyChecker(Tester):
    def test(self):
        cnt1, cnt2, cnt3, cnt4 = 0, 0, 0, 0

        for data in self.dataloader:
            sharp_frames = data['sharp_frames']
            data1 = {
                'sharp_frames': [sharp_frames[0], sharp_frames[-1]]
            }
            data2 = {
                'sharp_frames': [sharp_frames[1], sharp_frames[-2]]
            }
            data3 = {
                'sharp_frames': [sharp_frames[2], sharp_frames[-3]]
            }

            with torch.no_grad():
                d00, d01 = self.model(data1)
                u0 = self.model.hyperplane(d00).item()
                v0 = self.model.hyperplane(d01).item()

                d10, d11 = self.model(data2)
                u1 = self.model.hyperplane(d10).item()
                # v1 = self.model.hyperplane(d11).item()

                d20, d21 = self.model(data3)
                u2 = self.model.hyperplane(d20).item()
                # v2 = self.model.hyperplane(d21).item()

            s1 = -1 if u0 < 0 else 1
            s2 = -1 if u1 < 0 else 1
            s3 = -1 if u2 < 0 else 1

            cnt1 += (s1 == s2 and s2 == s3)
            cnt2 += (s1 == -1)
            cnt3 += (u0 * v0 < 0)
            cnt4 += 1

            if cnt3 == 5000:
                break

            if cnt4 % 100 == 0:
                print(cnt1, cnt2, cnt3, cnt4)
