import torch
import torch.nn as nn


class gLN(nn.Module):

    def __init__(self, dim):
        super(gLN, self).__init__()
        self.gamma = nn.Parameter(torch.ones([1, dim, 1, 1]))
        self.beta = nn.Parameter(torch.zeros([1, dim, 1, 1]))

    def forward(self, input):  # input :[batch_size, channels, time_step]
        batch_size, channels, fre, time = input.shape
        mean = input.mean(dim=[1, 2, 3], keepdim=True)
        var = (input - mean).pow(2).mean(dim=[1, 2, 3], keepdim=True)
        return ((input - mean) / (var + 1e-8).sqrt()) * self.gamma + self.beta


class InformationComm(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(InformationComm, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1))

    def forward(self, input, output_ref):
        outputs = output_ref * torch.tanh(self.conv(input))
        return outputs


class FTB(nn.Module):
    """
       Frequency transformation block for amplitude
       Non-local correlations exist in T-F spectrogram along the frequency axis.
       T-F attention
    """

    def __init__(self, fre_dim=257, in_channel=9, Cr=5):
        super(FTB, self).__init__()

        # pointwise conv to get Sa
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, Cr, kernel_size=[1, 1]), nn.BatchNorm2d(Cr),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv1d(Cr * fre_dim, in_channel, kernel_size=9, padding=4),
            nn.BatchNorm1d(in_channel), nn.ReLU())

        # flatten Sa, use FreqFC to get Str, FTM(frequency transformation matrix)
        self.FreqFC = nn.Linear(fre_dim, fre_dim)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel, kernel_size=[1, 1]),
            nn.BatchNorm2d(in_channel), nn.ReLU())

    def forward(self, input):  # input [B, Channel, Fre, Time]
        Sa_tmp = self.conv1(input)  # [B, Cr, Fre, Time]
        shape = Sa_tmp.shape
        Sa_tmp = Sa_tmp.view(shape[0], shape[1] * shape[2],
                             shape[3])  # [B, Cr*Fre, Time]
        Sa = self.conv2(Sa_tmp)  # [B, Channel, Time]
        shape = Sa.shape
        Sa = Sa.view(shape[0], shape[1], 1, shape[2])  # [B, Channel, 1, Time]

        Sa = Sa * input  # [B, Channel, Fre, Time]

        Sa = Sa.permute(0, 1, 3, 2)  # [B, Channel, Time, Fre]
        Str = self.FreqFC(Sa)
        Str = Str.permute(0, 1, 3, 2)  # [B, Channel, Fre, Time]
        So = self.out_conv(torch.cat(
            [Str, input],
            1))  # [B, Channel*2, Fre, Time] -> [B, Channel, Fre, Time]
        return So


class TSB(nn.Module):

    def __init__(self, fre_dim, amp_channel=9, phase_channel=8):
        super(TSB, self).__init__()
        self.ftb_begin = FTB(fre_dim=fre_dim, in_channel=amp_channel, Cr=5)

        self.amp_conv = nn.Sequential(
            nn.Conv2d(amp_channel,
                      amp_channel,
                      kernel_size=[5, 5],
                      padding=(2, 2)), nn.BatchNorm2d(amp_channel), nn.ReLU(),
            nn.Conv2d(amp_channel,
                      amp_channel,
                      kernel_size=[1, 25],
                      padding=(0, 12)), nn.BatchNorm2d(amp_channel), nn.ReLU(),
            nn.Conv2d(amp_channel,
                      amp_channel,
                      kernel_size=[5, 5],
                      padding=(2, 2)), nn.BatchNorm2d(amp_channel), nn.ReLU())
        self.ftb_end = FTB(fre_dim=fre_dim, in_channel=amp_channel, Cr=5)

        # no activation function is used
        self.phase_conv = nn.Sequential(
            gLN(phase_channel),
            nn.Conv2d(phase_channel,
                      phase_channel,
                      kernel_size=[5, 5],
                      padding=(2, 2)), gLN(phase_channel),
            nn.Conv2d(phase_channel,
                      phase_channel,
                      kernel_size=[1, 25],
                      padding=(0, 12)))

        self.ic_amp2phase = InformationComm(amp_channel, phase_channel)
        self.ic_phase2amp = InformationComm(phase_channel, amp_channel)

    def forward(self, amp, phase):
        ftb_out = self.ftb_begin(amp)
        amp_out = self.amp_conv(ftb_out)
        amp_out = self.ftb_end(amp_out)

        phase_out = self.phase_conv(phase)
        amp_out_final = self.ic_phase2amp(phase_out, amp_out)
        phase_out_final = self.ic_amp2phase(amp_out, phase_out)
        return amp_out_final, phase_out_final


class BiLSTM(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)

    def forward(self, input):  # [B, S, input_size]
        batch_size = input.shape[0]
        hidden_state = torch.randn(1 * 2, batch_size, self.hidden_size).to(
            input.device
        )  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.randn(1 * 2, batch_size, self.hidden_size).to(
            input.device
        )  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        input = input.permute(1, 0, 2)
        output, _ = self.lstm(input, (hidden_state, cell_state))
        return output.permute(1, 0, 2)


class Phasen(nn.Module):

    def __init__(self,
                 fre_dim=257,
                 amp_channel=9,
                 phase_channel=8,
                 num_blocks=3,
                 rnn_units=256):
        super(Phasen, self).__init__()

        # Stream A, input complex spectrogram
        self.aconv = nn.Sequential(
            nn.Conv2d(2, amp_channel, kernel_size=[1, 7], padding=(0, 3)),
            nn.BatchNorm2d(amp_channel), nn.ReLU(),
            nn.Conv2d(amp_channel,
                      amp_channel,
                      kernel_size=[7, 1],
                      padding=(3, 0)), nn.BatchNorm2d(amp_channel), nn.ReLU())

        # Stream P
        self.pconv = nn.Sequential(
            nn.Conv2d(2, phase_channel, kernel_size=[3, 5], padding=(1, 2)),
            nn.BatchNorm2d(phase_channel), nn.ReLU(),
            nn.Conv2d(phase_channel,
                      phase_channel,
                      kernel_size=[1, 25],
                      padding=(0, 12)), nn.BatchNorm2d(phase_channel),
            nn.ReLU())

        self.tsb_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.tsb_blocks.append(TSB(fre_dim, amp_channel, phase_channel))

        self.aconv1 = nn.Sequential(
            nn.Conv2d(amp_channel, 8, kernel_size=[1, 1]), nn.BatchNorm2d(8),
            nn.ReLU())
        self.ablstm = BiLSTM(8 * fre_dim, rnn_units)
        self.afc = nn.Sequential(nn.Linear(rnn_units * 2, 600), nn.ReLU(),
                                 nn.Linear(600, 600), nn.ReLU(),
                                 nn.Linear(600, 514 // 2), nn.Sigmoid())

        self.pconv1 = nn.Conv1d(phase_channel, 2, kernel_size=[1, 1])

    def forward(self, input):  # [B, 2, Fre, Time]
        amp_pre = self.aconv(input)
        phase_pre = self.pconv(input)

        for i, tsb in enumerate(self.tsb_blocks):
            if i > 0:
                amp_pre = amp_pre + amp
                phase_pre = phase_pre + phase
            amp, phase = tsb(amp_pre, phase_pre)

        amp = self.aconv1(amp)  # [B, 8, Fre, Time]

        #BiLSTM
        shape = amp.shape
        amp = amp.permute(0, 3, 2, 1).reshape(
            shape[0], shape[3],
            shape[1] * shape[2])  # [B, Time, Fre, 8] -> [B, Time, Fre*8]
        amp = self.ablstm(amp)
        amp = self.afc(amp)  # [B, Time, 514]
        amp = amp.unsqueeze(-1).permute(0, 3, 2, 1)  # [B, 1, Fre, Time]

        phase = self.pconv1(phase)
        phase = phase / torch.norm(phase, p=2, dim=1,
                                   keepdim=True)  # [B, 2, Fre, Time]
        amp_feat = torch.sqrt(input.pow(2).mean(1, keepdim=True))
        return amp_feat * amp * phase


def test_phase():
    stft_feat = torch.randn([1, 2, 257, 100])
    phase = Phasen(257)
    feat_out = phase(stft_feat)
